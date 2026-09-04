"""
RAW 解码层: 基于 LibRaw (rawpy), 输出线性光 float32。

与常规做法的三点不同 (决定最终画质):
    1. 全程在线性域运算, 不做 gamma / 自动亮度, 避免降噪在编码域失效
    2. 相机原生色彩空间输出 (ColorSpace.raw), 白平衡延后到色彩模块统一控制
    3. 保留传感器元数据 (黑/白电平、光学遮罩区), 供噪声剖面标定使用
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np

__all__ = ["DecodedImage", "decode_raw", "decode_image", "decode", "read_exif", "DemosaicName", "DEMOSAIC_ALGOS"]

try:
    import rawpy
    HAS_RAWPY = True
except Exception:  # pragma: no cover
    rawpy = None
    HAS_RAWPY = False


class DemosaicName:
    LINEAR = "linear"
    VNG = "vng"
    PPG = "ppg"
    AHD = "ahd"
    DCB = "dcb"
    AMAZE = "amaze"
    DHT = "dht"
    AAHD = "aahd"
    LMMSE = "lmmse"


# 数值索引 (LibRaw 常量顺序), 未装 rawpy 时仍可索引
DEMOSAIC_ALGOS = {
    DemosaicName.LINEAR: 0,
    DemosaicName.VNG: 1,
    DemosaicName.PPG: 2,
    DemosaicName.AHD: 3,
    DemosaicName.DCB: 4,
}

if HAS_RAWPY:
    _ahd = rawpy.DemosaicAlgorithm.AHD

    def _algo(name):
        return getattr(rawpy.DemosaicAlgorithm, name, _ahd)

    # 有 rawpy 时全部替换为枚举对象 (rawpy 会对枚举做有效性校验, 传 int 会报错)
    for _key, _name in (
        (DemosaicName.LINEAR, "LINEAR"),
        (DemosaicName.VNG, "VNG"),
        (DemosaicName.PPG, "PPG"),
        (DemosaicName.AHD, "AHD"),
        (DemosaicName.DCB, "DCB"),
        (DemosaicName.AMAZE, "AMAZE"),
        (DemosaicName.DHT, "DHT"),
        (DemosaicName.AAHD, "AAHD"),
        (DemosaicName.LMMSE, "LMMSE"),
    ):
        DEMOSAIC_ALGOS[_key] = _algo(_name)


@dataclass
class DecodedImage:
    """解码结果容器。

    属性:
        linear:   HxWx3 float32, 线性光, 相机原生色彩空间, 已归一化到 ~0..1
        is_raw:   是否来自 RAW
        path:     源文件路径
        width / height: 像素尺寸
        meta:     元数据字典 (相机/镜头/ISO/黑电平等)
        cam_xyz:  相机 RGB -> XYZ 矩阵 (3x3), 普通图为 None
        camera_wb: 相机白平衡乘数 (RGBG)
    """

    linear: np.ndarray
    is_raw: bool
    path: str
    width: int = 0
    height: int = 0
    meta: dict = field(default_factory=dict)
    cam_xyz: Optional[np.ndarray] = None
    camera_wb: Optional[np.ndarray] = None
    daylight_wb: Optional[np.ndarray] = None   # 矩阵标定光源(日光)下的白平衡, 白平衡计算必需
    black_level: float = 0.0
    white_level: float = 65535.0
    sensor: Optional[np.ndarray] = None     # 拜耳原始阵列 (含光学遮罩区), 供噪声标定
    sizes: dict = field(default_factory=dict)

    def __post_init__(self):
        self.height, self.width = self.linear.shape[:2]

    @property
    def size(self) -> Tuple[int, int]:
        return (self.width, self.height)


# --------------------------------------------------------------------------
# EXIF
# --------------------------------------------------------------------------

def _exif_from_bytes(data) -> dict:
    """从内存字节流 (JPEG/TIFF) 提取 EXIF 关键字段。"""
    import io as _io
    out = {}
    try:
        import exifread
        tags = exifread.process_file(_io.BytesIO(data), details=False)
    except Exception:
        return out

    def _s(key):
        v = tags.get(key)
        return str(v).strip() if v is not None else None

    out["make"] = _s("Image Make")
    out["model"] = _s("Image Model")
    out["lens_model"] = _s("EXIF LensModel") or _s("EXIF LensInfo")
    out["iso"] = _s("EXIF ISOSpeedRatings") or _s("EXIF PhotographicSensitivity")
    out["exposure"] = _s("EXIF ExposureTime")
    out["fnumber"] = _s("EXIF FNumber")
    out["focal"] = _s("EXIF FocalLength")
    out["datetime"] = _s("EXIF DateTimeOriginal")
    out["software"] = _s("Image Software")
    try:
        fl = str(out.get("focal") or "")
        if fl:
            out["focal_mm"] = float(eval(fl)) if "/" in fl else float(fl)  # noqa: S307
    except Exception:
        pass
    try:
        fn = str(out.get("fnumber") or "")
        if fn:
            out["fnumber_val"] = float(eval(fn)) if "/" in fn else float(fn)  # noqa: S307
    except Exception:
        pass
    return out


def read_exif(path) -> dict:
    """读取 EXIF 关键字段 (供镜头识别使用), 失败返回空字典。"""
    try:
        with open(str(path), "rb") as fh:
            head = fh.read(16)
        # ISO BMFF 容器 (CR3/HEIC 等) 文件头: 前4字节为 box 长度, 偏移4-7为 "ftyp"。
        # exifread 无法解析, 且会向 stdout 打印 "File format not recognized."。
        # 直接跳过, 由 decode() 内的缩略图 fallback 补齐 EXIF。
        if head[4:8] == b"ftyp":
            return {}
        with open(str(path), "rb") as fh:
            return _exif_from_bytes(fh.read())
    except Exception:
        return {}


def _merge_meta(base: dict, extra: dict) -> dict:
    """用 extra 补齐 base 中的空字段 (保留已有值)。"""
    out = dict(base)
    for k, v in extra.items():
        if v and (k not in out or not out[k]):
            out[k] = v
    return out


# --------------------------------------------------------------------------
# RAW 解码
# --------------------------------------------------------------------------

def _postprocess_compat(raw, params: dict):
    """调用 LibRaw 后处理, 自动兼容不同 rawpy 版本的参数差异。

    各版本 rawpy 的 Params 字段有增删 (如 0.27 移除了 gamma_16bit)。
    这里用"报错即剔除"的降级重试, 保证在 0.18~0.27 全系可用。
    """
    p = dict(params)
    for _ in range(12):
        try:
            return raw.postprocess(**p)
        except TypeError as e:
            msg = str(e)
            key = None
            if "unexpected keyword argument" in msg:
                key = msg.split("'")[1] if "'" in msg else None
            if key and key in p:
                p.pop(key)
                continue
            raise
    return raw.postprocess(**p)


def decode_raw(
    path,
    demosaic: str = DemosaicName.AHD,
    half_size: bool = False,
    highlight_mode: str = "blend",
    use_camera_wb: bool = False,
    fbdd_nr: bool = False,
    median_passes: int = 0,
) -> DecodedImage:
    """解码 RAW 到线性 float32 (相机原生色彩空间)。"""
    if not HAS_RAWPY:
        raise RuntimeError("rawpy 未安装, 无法解码 RAW。请运行 setup_env.bat 安装依赖。")

    meta = read_exif(path)
    algo = DEMOSAIC_ALGOS.get(demosaic, rawpy.DemosaicAlgorithm.AHD)

    hl_mode = {
        "clip": rawpy.HighlightMode.Clip,
        "blend": rawpy.HighlightMode.Blend,
        "reconstruct": getattr(rawpy.HighlightMode, "ReconstructDefault", rawpy.HighlightMode.Blend),
    }.get(highlight_mode, rawpy.HighlightMode.Blend)

    with rawpy.imread(str(path)) as raw:
        sizes = raw.sizes
        try:
            cam_wb = np.asarray(raw.camera_whitebalance, dtype=np.float32)
        except Exception:
            cam_wb = None
        try:
            day_wb = np.asarray(raw.daylight_whitebalance, dtype=np.float32)
        except Exception:
            day_wb = None
        try:
            black = np.asarray(raw.black_level_per_channel, dtype=np.float32)
            black_val = float(np.median(black[black > 0])) if np.any(black > 0) else 0.0
        except Exception:
            black_val = 0.0
        try:
            white_val = float(np.max(np.asarray(raw.white_level, dtype=np.float32)))
        except Exception:
            white_val = 65535.0

        # 传感器原始值 (含光学遮罩区), 供噪声标定
        sensor = None
        try:
            sensor = raw.raw_image.copy()
        except Exception:
            sensor = None

        params = dict(
            demosaic_algorithm=algo,
            half_size=half_size,
            four_color_rgb=False,
            fbdd_noise_reduction=(
                rawpy.FBDDNoiseReductionMode.Full if fbdd_nr
                else rawpy.FBDDNoiseReductionMode.Off
            ),
            noise_thr=None,
            median_filter_passes=median_passes,
            use_camera_wb=use_camera_wb,
            use_auto_wb=False,
            user_wb=[1.0, 1.0, 1.0, 1.0],
            output_color=rawpy.ColorSpace.raw,
            output_bps=16,
            gamma=(1.0, 1.0),          # 线性输出
            no_auto_bright=True,        # 不做自动亮度
            auto_bright_thr=None,
            highlight_mode=hl_mode,
            exp_shift=None,
            exp_preserve_highlights=0.0,
            no_auto_scale=False,        # 自动缩放到 0..65535
            gamma_16bit=False,
        )
        rgb16 = _postprocess_compat(raw, params)

        # 相机 -> XYZ 矩阵
        #
        # rawpy 的 rgb_xyz_matrix 形状为 (4,3), 前 3 行构成 3x3 矩阵, 语义是
        # **XYZ -> 相机 RGB** (按绿通道归一化), 即:
        #       cam_rgb = M @ xyz
        # 因此相机 RGB -> XYZ 需要求逆:
        #       xyz = inv(M) @ cam_rgb
        # 常见误解: 把它当成 cam->XYZ 直接用, 会得到严重偏色。
        cam_xyz = None
        try:
            m = np.asarray(raw.rgb_xyz_matrix, dtype=np.float64)
            m3 = m[:3, :3] if m.shape[0] >= 3 and m.shape[1] >= 3 else None
            if m3 is not None and abs(float(np.linalg.det(m3))) > 1e-12:
                cam_xyz = np.linalg.inv(m3).astype(np.float32)
        except Exception:
            cam_xyz = None

        meta.update({
            "raw_width": getattr(sizes, "width", 0),
            "raw_height": getattr(sizes, "height", 0),
            "raw_type": str(getattr(raw, "raw_type", "")),
            "num_colors": getattr(raw, "num_colors", 3),
            "color_desc": str(getattr(raw, "color_desc", "")),
        })

        # CR3 等 ISO BMFF 容器的 EXIF 无法被通用读取器解析, 但内嵌缩略图
        # (JPEG) 携带完整 EXIF。这里用它补齐缺失的机身/镜头/ISO 字段。
        if not (meta.get("make") and meta.get("model")):
            try:
                thumb = raw.extract_thumb()
                if thumb is not None and getattr(thumb, "data", None):
                    meta = _merge_meta(meta, _exif_from_bytes(thumb.data))
            except Exception:
                pass

        # rawpy 的 LibRaw 能直接从 MakerNote 解析镜头型号 (缩略图 EXIF 常缺失该 tag)。
        # 例如 CR3 里 raw.lens.model == "RF70-200mm F2.8 L IS USM"。
        if not meta.get("lens_model"):
            try:
                lens_obj = getattr(raw, "lens", None)
                if lens_obj is not None and getattr(lens_obj, "model", None):
                    meta["lens_model"] = str(lens_obj.model).strip()
            except Exception:
                pass

    linear = rgb16.astype(np.float32) / 65535.0
    linear = np.clip(linear, 0.0, None)

    return DecodedImage(
        linear=linear,
        is_raw=True,
        path=str(path),
        meta=meta,
        cam_xyz=cam_xyz,
        camera_wb=cam_wb,
        daylight_wb=day_wb,
        black_level=black_val,
        white_level=white_val,
        sensor=sensor,
        sizes={
            "top": getattr(sizes, "top", 0),
            "left": getattr(sizes, "left", 0),
            "iwidth": getattr(sizes, "iwidth", 0),
            "iheight": getattr(sizes, "iheight", 0),
            "width": getattr(sizes, "width", 0),
            "height": getattr(sizes, "height", 0),
        },
    )


# --------------------------------------------------------------------------
# 普通图像解码
# --------------------------------------------------------------------------

def decode_image(path) -> DecodedImage:
    """读取 JPEG/PNG/TIFF 等普通图像, 转为线性光 RGB。"""
    from .color import srgb_to_linear
    from . import io_utils

    bgr = io_utils.imread(path, cv2.IMREAD_UNCHANGED)
    if bgr is None:
        raise RuntimeError(f"无法读取图像: {path}")

    # 统一成 3 通道 BGR
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    elif bgr.shape[2] == 4:
        bgr = bgr[:, :, :3]

    scale = 65535.0 if bgr.dtype == np.uint16 else 255.0
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / scale
    linear = srgb_to_linear(rgb)

    meta = read_exif(path)
    return DecodedImage(
        linear=linear,
        is_raw=False,
        path=str(path),
        meta=meta,
        cam_xyz=None,
        camera_wb=None,
    )


def decode(path, **kwargs) -> DecodedImage:
    """自动分派: RAW 走 LibRaw, 普通图走 OpenCV。"""
    p = str(path)
    ext = os.path.splitext(p)[1].lower()
    if ext in (".dng", ".cr2", ".cr3", ".nef", ".arw", ".orf", ".rw2", ".raf",
               ".pef", ".srf", ".sr2", ".raw", ".x3f", ".3fr", ".erf", ".mrw",
               ".nrw", ".rwl", ".dcr", ".k25", ".kdc", ".mef", ".mos", ".mrw",
               ".ptx", ".r3d", ".fff", ".iiq", ".cap", ".dcs", ".drf", ".mdc",
               ".srw", ".tif", ".tiff"):
        # .tif 可能是线性 DNG, 先尝试 RAW
        try:
            return decode_raw(p, **kwargs)
        except Exception:
            if ext in (".tif", ".tiff"):
                return decode_image(p)
            raise
    return decode_image(p)
