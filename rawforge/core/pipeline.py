"""
主流水线编排: RAW -> 成片。

处理顺序 (每一步都在前一步的输出上工作, 顺序不可随意调换):

    1. 解码      LibRaw 去马赛克, 输出线性光相机原生 RGB
    2. 噪声标定  从光学遮罩区 + 图像分块统计拟合泊松-高斯剖面
    3. 镜头校正  畸变 / 色差 / 暗角 (必须在去马赛克后的线性域做)
    4. 白平衡    相机空间乘通道增益 (降噪前做, 否则蓝通道噪声比例失真)
    5. 色彩转换  相机 RGB -> XYZ -> 目标 RGB 线性值
    6. 降噪      亮度/色度分离, 噪声剖面自适应
    7. 渲染      曝光 / 高光滚降 / 饱和度 / gamma 编码
    8. 输出      16bit TIFF / PNG / JPEG / Linear DNG

两种工作模式:
    full        : 完整出片, 输出可直接使用的图像
    denoise_only: 只做解码+校正+降噪, 输出线性 DNG/16bit TIFF,
                  保留全部后期空间交给 Lightroom / Capture One / darktable
                  (这正是 DxO PureRAW 的核心用法)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional, List

import cv2
import numpy as np

from . import decode as decode_mod
from . import color as color_mod
from . import io_utils
from .noise import NoiseProfile, estimate_noise_profile, estimate_sensor_read_noise, profile_from_iso
from .lens import LensCorrectionParams, LensCorrector, LensInfo
from .denoise import DenoiseParams, denoise as denoise_fn, auto_params_from_profile

__all__ = ["PipelineParams", "ProcessResult", "RawPipeline", "process_file", "STAGES"]


STAGES = [
    ("decode", "解码"),
    ("profile", "噪声标定"),
    ("lens", "镜头校正"),
    ("color", "色彩转换"),
    ("denoise", "降噪"),
    ("render", "渲染"),
    ("write", "输出"),
]


@dataclass
class RenderParams:
    """渲染与输出参数。"""
    exposure_ev: float = 0.0          # 曝光补偿 (EV)
    contrast: float = 0.0             # -1..1, S 曲线对比度
    saturation: float = 1.0           # 0..2
    highlight_rolloff: float = 0.0    # 0..1, 高光压缩 (0=硬裁剪)
    shadow_lift: float = 0.0          # 0..1, 提亮暗部
    white_balance: str = "camera"     # camera / auto / manual / none
    manual_wb: Optional[list] = None  # [r,g,b]
    wb_temp: float = 5500.0           # manual 时的色温
    output_space: str = "srgb"        # srgb / adobe_rgb / prophoto / rec2020
    auto_exposure: bool = True        # 自动归一化白点

    def to_dict(self):
        return asdict(self)


@dataclass
class OutputParams:
    """输出参数。"""
    mode: str = "full"                # full / denoise_only
    format: str = "tiff16"            # tiff16 / png16 / png8 / jpg / dng16
    quality: int = 95                 # JPEG 质量
    output_dir: str = ""              # 空 = 源文件同目录
    suffix: str = "_rawforge"
    overwrite: bool = False

    def to_dict(self):
        return asdict(self)


@dataclass
class PipelineParams:
    """完整流水线参数。"""
    demosaic: str = "ahd"
    highlight_mode: str = "blend"
    half_size: bool = False
    lens: LensCorrectionParams = field(default_factory=LensCorrectionParams)
    denoise: DenoiseParams = field(default_factory=DenoiseParams)
    render: RenderParams = field(default_factory=RenderParams)
    output: OutputParams = field(default_factory=OutputParams)

    def to_dict(self):
        return {
            "demosaic": self.demosaic,
            "highlight_mode": self.highlight_mode,
            "half_size": self.half_size,
            "lens": self.lens.to_dict(),
            "denoise": self.denoise.to_dict(),
            "render": self.render.to_dict(),
            "output": self.output.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict):
        p = cls()
        if not d:
            return p
        for k in ("demosaic", "highlight_mode", "half_size"):
            if k in d:
                setattr(p, k, d[k])
        if "lens" in d:
            p.lens = LensCorrectionParams(**d["lens"])
        if "denoise" in d:
            p.denoise = DenoiseParams(**d["denoise"])
        if "render" in d:
            p.render = RenderParams(**d["render"])
        if "output" in d:
            p.output = OutputParams(**d["output"])
        return p


@dataclass
class ProcessResult:
    """单文件处理结果。"""
    source: str = ""
    output: str = ""
    ok: bool = False
    message: str = ""
    elapsed: float = 0.0
    profile: Optional[NoiseProfile] = None
    lens_info: Optional[LensInfo] = None
    width: int = 0
    height: int = 0

    def __str__(self):
        status = "成功" if self.ok else "失败"
        return f"[{status}] {os.path.basename(self.source)} -> {self.output or '-'} ({self.elapsed:.2f}s)"


class RawPipeline:
    """RAW 处理流水线。"""

    def __init__(self, params: Optional[PipelineParams] = None):
        self.params = params or PipelineParams()
        self.corrector = LensCorrector()

    # ------------------------------------------------------------------

    def process(self, path, progress: Optional[Callable[[str, float], None]] = None) -> ProcessResult:
        t0 = time.time()
        res = ProcessResult(source=str(path))
        P = self.params

        def emit(stage, pct):
            if progress:
                try:
                    progress(stage, pct)
                except Exception:
                    pass

        try:
            # ---------- 1. 解码 ----------
            emit("decode", 0.02)
            dec = decode_mod.decode(
                path,
                demosaic=P.demosaic,
                half_size=P.half_size,
                highlight_mode=P.highlight_mode,
                use_camera_wb=False,
            )
            img = dec.linear
            res.width, res.height = dec.width, dec.height
            emit("decode", 0.12)

            # ---------- 2. 噪声标定 ----------
            emit("profile", 0.15)
            read_noise = None
            if dec.is_raw and dec.sensor is not None:
                read_noise = estimate_sensor_read_noise(
                    dec.sensor, dec.black_level, dec.white_level, dec.sizes
                )
            wb_for_profile = dec.camera_wb[:3] if (dec.camera_wb is not None and dec.camera_wb.size >= 3) else None
            profile = estimate_noise_profile(img, wb_gains=wb_for_profile, read_noise=read_noise)
            if profile.confidence < 0.05 and not read_noise:
                iso = 0.0
                try:
                    iso = float(str(dec.meta.get("iso") or "0").split()[0])
                except Exception:
                    iso = 0.0
                if iso > 0:
                    profile = profile_from_iso(iso)
            res.profile = profile
            emit("profile", 0.22)

            # ---------- 3. 镜头校正 ----------
            emit("lens", 0.25)
            lens_info = self.corrector.identify(dec.meta, P.lens)
            if P.lens.enable_distortion or P.lens.enable_tca or P.lens.enable_vignetting:
                img, lens_info = self.corrector.correct(img, dec.meta, P.lens, lens_info)
            res.lens_info = lens_info
            emit("lens", 0.42)

            # ---------- 4. 白平衡 ----------
            #
            # 关键: LibRaw 的 rgb_xyz_matrix 期望输入是**日光白平衡**下的相机 RGB。
            #       因此相机空间的白平衡增益必须写为 pre_mul = wb_gains / daylight_wb,
            #       否则 (尤其钨丝灯场景) 会整体偏色。
            emit("color", 0.45)
            R = P.render
            day_wb = dec.daylight_wb
            if day_wb is None or np.asarray(day_wb).size < 3 or not np.any(np.asarray(day_wb) > 0):
                day_wb = None

            if R.white_balance == "camera" and dec.camera_wb is not None:
                wb_gains = np.asarray(dec.camera_wb, dtype=np.float32)[:3]
            elif R.white_balance == "auto":
                wb_gains = color_mod.gray_world_wb(img)
            elif R.white_balance == "manual" and R.manual_wb:
                wb_gains = np.asarray(R.manual_wb, dtype=np.float32)[:3]
            elif R.white_balance == "temp":
                # 色温模式走 Bradford 色适应 (在 XYZ 空间), 相机空间增益按日光处理
                wb_gains = np.asarray(day_wb if day_wb is not None
                                      else (dec.camera_wb[:3] if dec.camera_wb is not None
                                            else [1.0, 1.0, 1.0]), dtype=np.float32)
            else:
                # none: 相当于日光白平衡 (不做额外调整)
                wb_gains = np.asarray(day_wb if day_wb is not None else [1.0, 1.0, 1.0],
                                      dtype=np.float32)

            wb_gains = np.asarray(wb_gains, dtype=np.float32)[:3]
            if day_wb is not None:
                pre_mul = wb_gains / np.asarray(day_wb, dtype=np.float32)[:3]
            else:
                pre_mul = wb_gains
            # 归一化保持整体亮度 (几何平均), 不改变通道比例
            pre_mul = pre_mul / float(np.exp(np.mean(np.log(np.maximum(pre_mul, 1e-6)))))
            img = img * pre_mul.reshape(1, 1, 3)
            emit("color", 0.52)

            # ---------- 5. 色彩空间转换 ----------
            if P.output.mode == "denoise_only":
                # 保留相机原生空间, 不渲染、不转色彩空间
                emit("denoise", 0.55)
                sigma_map = profile.sigma(np.clip(img.mean(axis=2), 0.0, None))
                dn_params = P.denoise
                img = denoise_fn(img, profile, dn_params, sigma_map)
                emit("denoise", 0.85)
                out_img = np.clip(img, 0.0, None).astype(np.float32)
                emit("render", 0.9)
            else:
                cam_xyz = dec.cam_xyz
                if cam_xyz is not None:
                    xyz = img @ cam_xyz.T
                    # 手动色温: Bradford 色适应 (D65 中性 -> 目标色温)
                    if R.white_balance == "temp":
                        xyz = color_mod.adapt_xyz(
                            xyz, color_mod.D65_XY, color_mod.kelvin_to_xy(R.wb_temp))
                    linear_out = color_mod.xyz_to_rgb(xyz, R.output_space)
                else:
                    linear_out = img
                emit("color", 0.58)

                # 曝光
                if R.exposure_ev:
                    linear_out = linear_out * np.float32(2.0 ** R.exposure_ev)
                if R.auto_exposure:
                    linear_out = color_mod.normalize_white(linear_out, 99.9)

                # ---------- 6. 降噪 ----------
                emit("denoise", 0.60)
                sigma_map = profile.sigma(np.clip(linear_out.mean(axis=2), 0.0, None))
                img_dn = denoise_fn(linear_out, profile, P.denoise, sigma_map)
                # 统一 NaN/Inf 保护: 镜头校正 remap + 大尺寸降噪在边缘可能产生
                # 异常值, 进入色调前必须清理, 否则会出现黑色斑点或过曝伪影。
                img_dn = np.nan_to_num(img_dn, nan=0.0, posinf=4.0, neginf=0.0).astype(np.float32)
                img_dn = np.clip(img_dn, 0.0, 16.0).astype(np.float32)
                emit("denoise", 0.86)

                # ---------- 7. 渲染 ----------
                emit("render", 0.88)
                out_linear = self._tone(img_dn, R)
                encoded = color_mod.apply_transfer(out_linear, R.output_space)
                out_linear_final = encoded
                emit("render", 0.95)
                out_img = out_linear_final

            # ---------- 8. 输出 ----------
            emit("write", 0.96)
            out_path = self._write(path, out_img, P.output, dec)
            res.output = out_path
            res.ok = True
            res.elapsed = time.time() - t0
            emit("write", 1.0)
            return res

        except Exception as e:
            res.ok = False
            res.message = f"{type(e).__name__}: {e}"
            res.elapsed = time.time() - t0
            return res

    # ------------------------------------------------------------------

    def _tone(self, linear: np.ndarray, R: RenderParams) -> np.ndarray:
        """色调渲染: 高光滚降 / 暗部提亮 / 对比度 / 饱和度。"""
        x = np.clip(linear, 0.0, None).astype(np.float32)

        # 高光滚降: 用软化压缩替代硬裁剪, 保住云层/高光的层次
        if R.highlight_rolloff > 0:
            t = float(np.clip(R.highlight_rolloff, 0.0, 1.0))
            knee = 0.85
            hi = np.maximum(x - knee, 0.0)
            x = np.where(x > knee,
                         knee + (1.0 - knee) * (1.0 - np.exp(-hi / ((1.0 - knee) * (1.0 - t * 0.95 + 1e-6)))),
                         x).astype(np.float32)

        # 暗部提亮 (不影响黑点)
        if R.shadow_lift > 0:
            s = float(np.clip(R.shadow_lift, 0.0, 1.0))
            x = np.power(np.maximum(x, 0.0), 1.0 / (1.0 + s * 0.9)).astype(np.float32)

        # 对比度 S 曲线
        if abs(R.contrast) > 1e-3:
            c = float(np.clip(R.contrast, -1.0, 1.0))
            if c >= 0:
                x = np.clip(x, 0.0, 1.0)
                x = x + c * (x - 3 * x ** 2 + 2 * x ** 3) * 0.8
            else:
                x = np.clip(x, 0.0, 1.0)
                x = x + c * (x - 0.5) * 0.6
            x = np.clip(x, 0.0, None).astype(np.float32)

        # 饱和度
        if abs(R.saturation - 1.0) > 1e-3:
            lum = (0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2])[..., None]
            x = np.clip(lum + (x - lum) * float(R.saturation), 0.0, None).astype(np.float32)

        return np.clip(x, 0.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------

    def _write(self, src_path, img_float: np.ndarray, O: OutputParams, dec) -> str:
        """写出结果文件。"""
        base = os.path.splitext(os.path.basename(str(src_path)))[0]
        outdir = O.output_dir or os.path.dirname(os.path.abspath(str(src_path)))
        io_utils.ensure_dir(outdir)

        mode = O.mode
        fmt = O.format.lower()

        if mode == "denoise_only" and fmt not in ("dng16", "tiff16"):
            fmt = "tiff16"

        name = f"{base}{O.suffix}"
        if fmt == "jpg":
            ext = ".jpg"
        elif fmt == "dng16":
            ext = ".dng"
        elif fmt in ("png8", "png16"):
            ext = ".png"
        else:
            ext = ".tif"

        out_path = os.path.join(outdir, name + ext)
        if not O.overwrite:
            out_path = io_utils.unique_path(out_path)

        if fmt == "dng16":
            ok = self._write_linear_dng(out_path, img_float, dec)
            if ok:
                return out_path
            # 回退 TIFF
            out_path = os.path.splitext(out_path)[0] + ".tif"
            fmt = "tiff16"

        if fmt == "jpg":
            data8 = (np.clip(img_float, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
            # JPEG 用 Pillow 保存: 原生支持中文路径, subsampling=0 即 4:4:4 无色度抽样
            from PIL import Image
            Image.fromarray(data8, "RGB").save(
                out_path, "JPEG", quality=int(O.quality), subsampling=0, optimize=True)
        elif fmt == "png8":
            data8 = (np.clip(img_float, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
            bgr = cv2.cvtColor(data8, cv2.COLOR_RGB2BGR)
            io_utils.imwrite(out_path, bgr, [cv2.IMWRITE_PNG_COMPRESSION, 6])
        elif fmt == "png16":
            data16 = (np.clip(img_float, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
            bgr = cv2.cvtColor(data16, cv2.COLOR_RGB2BGR)
            io_utils.imwrite(out_path, bgr, [cv2.IMWRITE_PNG_COMPRESSION, 6])
        else:  # tiff16
            data16 = (np.clip(img_float, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
            bgr = cv2.cvtColor(data16, cv2.COLOR_RGB2BGR)
            io_utils.imwrite(out_path, bgr, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
        return out_path

    @staticmethod
    def _write_linear_dng(path: str, linear: np.ndarray, dec) -> bool:
        """写出 16bit 线性 DNG (去马赛克后的 LinearRaw)。

        DNG 本质是 TIFF 的特殊化, 关键是写入 DNGVersion、颜色矩阵、
        AsShotNeutral 这几个 tag。已去马赛克的数据用 LinearRaw 光度解释,
        不带 CFAPattern, Lightroom/Capture One/darktable 均可读入。
        """
        try:
            import tifffile
        except Exception:
            return False
        try:
            data = np.clip(linear, 0.0, 1.0)
            data16 = (data * 65535.0 + 0.5).astype(np.uint16)

            def rational(v, den=10000):
                n = int(round(float(v) * den))
                return (n, den)

            # ColorMatrix1: XYZ(D50) -> camera RGB 的逆矩阵, 这里写 camera->XYZ
            cam_xyz = dec.cam_xyz
            if cam_xyz is not None:
                m = np.asarray(cam_xyz, dtype=np.float64)
                flat = [rational(v) for v in m.flatten().tolist()]
            else:
                flat = [rational(v) for v in np.eye(3, dtype=float).flatten().tolist()]

            wb = dec.camera_wb
            if wb is not None and np.asarray(wb).size >= 3:
                w = np.asarray(wb, dtype=np.float64)[:3]
                w = w / max(float(w.max()), 1e-6)
                as_shot = [rational(v) for v in w.tolist()]
            else:
                as_shot = [rational(1.0), rational(1.0), rational(1.0)]

            extratags = [
                (50706, 1, 4, (1, 4, 0, 0), True),        # DNGVersion 1.4.0.0
                (50707, 1, 4, (1, 1, 0, 0), True),        # DNGBackwardVersion
                (50708, 2, 0, str(dec.meta.get("model") or "Unknown"), True),  # UniqueCameraModel
                (50721, 10, 9, tuple(flat), True),        # ColorMatrix1 (SRATIONAL)
                (50728, 10, 3, tuple(as_shot), True),     # AsShotNeutral
                (50730, 10, 1, (0, 1), True),             # BaselineExposure 0
                (50714, 4, 1, (0,), True),                # BlackLevel 0
                (50717, 4, 1, (65535,), True),            # WhiteLevel
                (33434, 4, 1, (1, 1), False),             # ExposureTime (避免某些阅读器报错)
            ]

            tifffile.imwrite(
                path,
                data16,
                photometric="linear_raw",
                planarconfig="contig",
                compression=None,
                extratags=extratags,
            )
            return True
        except Exception:
            return False


# --------------------------------------------------------------------------

def process_file(path, params: Optional[PipelineParams] = None,
                 progress: Optional[Callable[[str, float], None]] = None) -> ProcessResult:
    """便捷函数: 处理单个文件。"""
    return RawPipeline(params).process(path, progress)


def process_batch(paths, params: Optional[PipelineParams] = None,
                  workers: int = 1,
                  progress: Optional[Callable[[str, float, int, int], None]] = None) -> List[ProcessResult]:
    """批量处理。workers>1 时使用线程池 (numpy/OpenCV 运算会释放 GIL)。"""
    results = []
    total = len(paths)
    if workers <= 1 or total <= 1:
        for i, p in enumerate(paths):
            r = process_file(p, params,
                             (lambda s, v, i=i: progress(s, v, i, total)) if progress else None)
            results.append(r)
        return results

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = {ex.submit(process_file, p, params, None): p for p in paths}
        done = 0
        for fu in as_completed(futures):
            try:
                r = fu.result()
            except Exception as e:  # pragma: no cover
                r = ProcessResult(source=str(futures[fu]), ok=False, message=str(e))
            results.append(r)
            done += 1
            if progress:
                try:
                    progress("batch", done / max(total, 1), done, total)
                except Exception:
                    pass
    return results
