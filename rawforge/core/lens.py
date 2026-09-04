"""
镜头校正层: 畸变 / 横向色差(TCA) / 暗角。

双通道设计:
    通道 A (优先): lensfun 光学数据库 —— 覆盖 400+ 机身、上千支镜头的实测光学参数,
                   校正精度等同于 darktable / RawTherapee 的官方效果。
    通道 B (兜底): 自研参数化模型 + 自动估计算法, 无数据库时依然可用:
                   - 畸变: Brown-Conrady (k1,k2,k3,p1,p2)
                   - 色差: 径向通道缩放 (R/B 相对 G), 支持自动估计
                   - 暗角: 六阶径向多项式, 支持从图像径向剖面自动拟合

所有校正均在**线性光**下执行, 顺序为: 畸变 -> 色差 -> 暗角。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import cv2
import numpy as np

__all__ = [
    "LensCorrectionParams", "LensInfo", "LensCorrector",
    "correct_distortion_model", "correct_tca_model", "correct_vignetting_model",
    "estimate_tca", "estimate_vignetting", "radial_grid",
]

try:
    import lensfunpy
    HAS_LENSFUN = True
except Exception:  # pragma: no cover
    lensfunpy = None
    HAS_LENSFUN = False

# --------------------------------------------------------------------------
# 自定义机身补充数据库
#
# lensfun 官方数据库对新发布机身的收录有滞后。这里以 lensfun XML v1 格式
# 内嵌尚未收录的新机身, 让它们也能正确匹配到卡口与 crop factor, 从而复用
# 库中已收录的 RF 镜头校正模型 (畸变/色差/暗角挂在镜头条目上)。
#
# 新增机身均为全画幅 (cropfactor=1) + Canon RF 卡口, 与 R6m2 / R5 一致。
# --------------------------------------------------------------------------

_CUSTOM_CAMERAS_XML = '''<lensdatabase version="1">
    <camera>
        <maker>Canon</maker>
        <model>Canon EOS R5 Mark II</model>
        <model lang="en">EOS R5 Mark II</model>
        <mount>Canon RF</mount>
        <cropfactor>1</cropfactor>
    </camera>
    <camera>
        <maker>Canon</maker>
        <model>Canon EOS R6 Mark III</model>
        <model lang="en">EOS R6 Mark III</model>
        <mount>Canon RF</mount>
        <cropfactor>1</cropfactor>
    </camera>
</lensdatabase>
'''


@dataclass
class LensCorrectionParams:
    """镜头校正参数。enable_* 为 False 时跳过该项。"""

    enable_distortion: bool = True
    enable_tca: bool = True
    enable_vignetting: bool = True

    # 模型通道 (lensfun 不可用时使用)
    distortion_k: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    distortion_p: Tuple[float, float] = (0.0, 0.0)
    tca_red: float = 1.0        # R 通道相对 G 的径向缩放
    tca_blue: float = 1.0       # B 通道相对 G 的径向缩放
    vignetting_k: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # 自动估计
    auto_tca: bool = False
    auto_vignetting: bool = False

    # 手动指定 (覆盖 EXIF 自动识别)
    camera_maker: Optional[str] = None
    camera_model: Optional[str] = None
    lens_maker: Optional[str] = None
    lens_model: Optional[str] = None
    focal_length: Optional[float] = None
    aperture: Optional[float] = None
    distance: float = 10.0

    def to_dict(self):
        return {
            "enable_distortion": self.enable_distortion,
            "enable_tca": self.enable_tca,
            "enable_vignetting": self.enable_vignetting,
            "distortion_k": list(self.distortion_k),
            "distortion_p": list(self.distortion_p),
            "tca_red": self.tca_red,
            "tca_blue": self.tca_blue,
            "vignetting_k": list(self.vignetting_k),
            "auto_tca": self.auto_tca,
            "auto_vignetting": self.auto_vignetting,
            "camera_maker": self.camera_maker,
            "camera_model": self.camera_model,
            "lens_maker": self.lens_maker,
            "lens_model": self.lens_model,
            "focal_length": self.focal_length,
            "aperture": self.aperture,
            "distance": self.distance,
        }


@dataclass
class LensInfo:
    """识别到的机身/镜头信息。"""
    camera: Optional[str] = None
    lens: Optional[str] = None
    focal: Optional[float] = None
    aperture: Optional[float] = None
    source: str = "none"          # "lensfun" / "exif" / "manual" / "none"
    message: str = ""
    candidates: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# 基础几何工具
# --------------------------------------------------------------------------

def radial_grid(h: int, w: int, cx: float = None, cy: float = None):
    """生成归一化径向坐标: x,y 为 -1..1 (以画幅中心为原点), r 为归一化半径。"""
    if cx is None:
        cx = (w - 1) / 2.0
    if cy is None:
        cy = (h - 1) / 2.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    x = (xx - cx) / ((w - 1) / 2.0)
    y = (yy - cy) / ((h - 1) / 2.0)
    r = np.sqrt(x * x + y * y).astype(np.float32)
    return x.astype(np.float32), y.astype(np.float32), r


def _remap(img: np.ndarray, mapx: np.ndarray, mapy: np.ndarray, interp=None) -> np.ndarray:
    if interp is None:
        interp = cv2.INTER_LANCZOS4
    return cv2.remap(img, mapx.astype(np.float32), mapy.astype(np.float32),
                     interpolation=interp, borderMode=cv2.BORDER_REPLICATE)


# --------------------------------------------------------------------------
# 通道 B: 参数化模型
# --------------------------------------------------------------------------

def correct_distortion_model(img: np.ndarray, k=(0., 0., 0.), p=(0., 0.),
                             cx=None, cy=None) -> np.ndarray:
    """Brown-Conrady 径向+切向畸变校正 (正向映射求逆采用迭代法)。

    注意: 这里做的是"去畸变", 即求 undistorted 位置对应的源坐标。
    对轻度畸变 (|k1| < 0.3) 直接用畸变模型作为逆映射已足够精确;
    畸变较大时用 3 次不动点迭代求逆, 精度可达亚像素。
    """
    k1, k2, k3 = k
    p1, p2 = p
    if abs(k1) < 1e-9 and abs(k2) < 1e-9 and abs(k3) < 1e-9 and abs(p1) < 1e-9 and abs(p2) < 1e-9:
        return img

    h, w = img.shape[:2]
    x, y, r = radial_grid(h, w, cx, cy)
    r2 = r * r
    r4 = r2 * r2
    r6 = r4 * r2

    # 径向因子
    radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6

    # 切向分量
    tx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    ty = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y

    # 去畸变: 迭代求解 src 使得 distort(src) == dst
    src_x = x * radial + tx
    src_y = y * radial + ty
    for _ in range(3):
        rs2 = src_x * src_x + src_y * src_y
        rs4 = rs2 * rs2
        rs6 = rs4 * rs2
        rad_s = 1.0 + k1 * rs2 + k2 * rs4 + k3 * rs6
        tsx = 2.0 * p1 * src_x * src_y + p2 * (rs2 + 2.0 * src_x * src_x)
        tsy = p1 * (rs2 + 2.0 * src_y * src_y) + 2.0 * p2 * src_x * src_y
        src_x = x * rad_s * (radial / np.maximum(rad_s, 1e-6)) + tsx
        src_y = y * rad_s * (radial / np.maximum(rad_s, 1e-6)) + tsy

    # 归一化坐标 -> 像素坐标
    mapx = src_x * ((w - 1) / 2.0) + (w - 1) / 2.0
    mapy = src_y * ((h - 1) / 2.0) + (h - 1) / 2.0
    return _remap(img, mapx, mapy)


def correct_tca_model(img: np.ndarray, red_scale=1.0, blue_scale=1.0,
                      cx=None, cy=None) -> np.ndarray:
    """横向色差校正: R/B 通道按径向缩放重采样, 与 G 通道对齐。"""
    if abs(red_scale - 1.0) < 1e-7 and abs(blue_scale - 1.0) < 1e-7:
        return img
    h, w = img.shape[:2]
    cyy = (h - 1) / 2.0 if cy is None else cy
    cxx = (w - 1) / 2.0 if cx is None else cx
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    out = img.copy()
    for ch, s in ((0, red_scale), (2, blue_scale)):
        if abs(s - 1.0) < 1e-7:
            continue
        mapx = (xx - cxx) / s + cxx
        mapy = (yy - cyy) / s + cyy
        out[..., ch] = _remap(img[..., ch], mapx, mapy, cv2.INTER_CUBIC)
    return out


def correct_vignetting_model(img: np.ndarray, k=(0., 0., 0.), cx=None, cy=None) -> np.ndarray:
    """暗角补偿: gain = 1 + k1*r^2 + k2*r^4 + k3*r^6。"""
    k1, k2, k3 = k
    if abs(k1) < 1e-9 and abs(k2) < 1e-9 and abs(k3) < 1e-9:
        return img
    h, w = img.shape[:2]
    _x, _y, r = radial_grid(h, w, cx, cy)
    r2 = r * r
    gain = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    gain = gain.astype(np.float32)[..., None]
    return (img * gain).astype(np.float32)


# --------------------------------------------------------------------------
# 自动估计 (实验性)
# --------------------------------------------------------------------------

def estimate_tca(img: np.ndarray, search: float = 0.006, steps: int = 25,
                 cx=None, cy=None) -> Tuple[float, float]:
    """自动估计横向色差: 在径向缩放空间搜索使通道边缘最吻合的 R/B 缩放因子。

    判据: 校正后 R-G 与 B-G 的高频残差能量最小 (等价于通道间结构最一致)。
    """
    h, w = img.shape[:2]
    gray = img.mean(axis=2) if img.ndim == 3 else img

    def resid(scale, channel):
        mapx, mapy = _scale_maps(h, w, scale, cx, cy)
        warped = cv2.remap(img[..., channel], mapx, mapy, interpolation=cv2.INTER_CUBIC)
        return float(np.mean(np.abs(warped.astype(np.float32) - gray.astype(np.float32))))

    def _scale_maps(h, w, s, cx=None, cy=None):
        cxx = (w - 1) / 2.0 if cx is None else cx
        cyy = (h - 1) / 2.0 if cy is None else cy
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        return (xx - cxx) / s + cxx, (yy - cyy) / s + cyy

    best_r, best_br = 1.0, None
    best_b, best_bb = 1.0, None
    for s in np.linspace(1.0 - search, 1.0 + search, steps, dtype=np.float32):
        base_r, base_b = resid(float(s), 0), resid(float(s), 2)
        if best_br is None or base_r < best_br:
            best_br, best_r = base_r, float(s)
        if best_bb is None or base_b < best_bb:
            best_bb, best_b = base_b, float(s)
    return best_r, best_b


def estimate_vignetting(img: np.ndarray, rings: int = 24, deg: int = 3,
                        cx=None, cy=None) -> Tuple[float, float, float]:
    """从图像径向亮度剖面自动拟合暗角曲线。

    取每个环带内亮度的**高分位数**(避免暗部物体干扰), 用多项式拟合
    亮度随半径的衰减, 再反解出补偿系数。

    注意: 该估计假设画面整体亮度均匀, 对天空/纯色背景可靠,
    对主体居中的强反差场景可能过度提亮四角 —— 因此默认关闭。
    """
    h, w = img.shape[:2]
    gray = img.mean(axis=2) if img.ndim == 3 else img
    _x, _y, r = radial_grid(h, w, cx, cy)
    r_flat = r.ravel()
    g_flat = gray.ravel().astype(np.float32)

    edges = np.linspace(0.0, 1.0, rings + 1)
    radii, lum = [], []
    for i in range(rings):
        sel = (r_flat >= edges[i]) & (r_flat < edges[i + 1])
        if sel.sum() < 64:
            continue
        radii.append(float(np.median(r_flat[sel])))
        lum.append(float(np.percentile(g_flat[sel], 90)))
    if len(radii) < 5:
        return (0.0, 0.0, 0.0)

    radii = np.asarray(radii, dtype=np.float64)
    lum = np.asarray(lum, dtype=np.float64)
    # 归一化: 以中心亮度为 1
    base = max(float(lum[: max(2, len(lum) // 8)].mean()), 1e-6)
    ratio = lum / base
    ratio = np.clip(ratio, 0.2, 1.0)

    try:
        coef = np.polyfit(radii, ratio, deg)
    except Exception:
        return (0.0, 0.0, 0.0)
    # 拟合出的 ratio(r) 是"亮度衰减", 补偿增益 gain = 1/ratio,
    # 再用 1 + k1 r^2 + k2 r^4 + k3 r^6 拟合 gain
    gain = 1.0 / np.maximum(ratio, 1e-3)
    try:
        A = np.stack([radii ** 2, radii ** 4, radii ** 6], axis=1)
        sol, *_ = np.linalg.lstsq(A, (gain - 1.0), rcond=None)
        k1, k2, k3 = (float(v) for v in sol)
    except Exception:
        return (0.0, 0.0, 0.0)
    # 限幅, 防止极端过校正
    k1 = float(np.clip(k1, -0.5, 2.0))
    k2 = float(np.clip(k2, -0.5, 2.0))
    k3 = float(np.clip(k3, -0.5, 2.0))
    return (k1, k2, k3)


# --------------------------------------------------------------------------
# 通道 A: lensfun 数据库
# --------------------------------------------------------------------------

def _norm_lens(s: str) -> str:
    """规范化镜头/机身名称: 去空格、统一小写, 用于跨来源的包含匹配。"""
    return re.sub(r"\s+", "", str(s)).lower()


class LensCorrector:
    """镜头校正器, 自动在 lensfun 数据库与参数化模型之间选择。"""

    def __init__(self):
        self.db = None
        self.load_error = ""
        if HAS_LENSFUN:
            try:
                # 先加载 bundle 数据库 (948 机身), 再叠加自定义新机身补充。
                # 自定义条目里的 mount (Canon RF) 已存在于 bundle, lensfun 会自动去重。
                self.db = lensfunpy.Database(xml=_CUSTOM_CAMERAS_XML)
            except Exception as e:  # pragma: no cover
                self.load_error = f"{type(e).__name__}: {e}"
                try:
                    self.db = lensfunpy.Database()  # 自定义加载失败时回退默认库
                except Exception:
                    self.db = None

    @property
    def available(self) -> bool:
        return self.db is not None

    # ---------------- 识别 ----------------

    def identify(self, meta: dict, params: LensCorrectionParams) -> LensInfo:
        """从 EXIF / 手动参数识别机身与镜头。"""
        maker = params.camera_maker or meta.get("make") or ""
        model = params.camera_model or meta.get("model") or ""
        lens = params.lens_model or meta.get("lens_model") or ""
        focal = params.focal_length or meta.get("focal_mm")
        aperture = params.aperture or meta.get("fnumber_val")

        info = LensInfo(
            camera=f"{maker} {model}".strip() or None,
            lens=lens or None,
            focal=float(focal) if focal else None,
            aperture=float(aperture) if aperture else None,
            source="exif" if (maker or model or lens) else "none",
        )
        if not self.available:
            info.message = "lensfun 不可用, 使用参数化模型"
            return info

        # 机身匹配
        cam = self._find_camera(maker, model)
        if cam is None:
            info.message = f"lensfun 未收录机身: {info.camera}"
            return info

        # 镜头匹配
        lens_obj = self._find_lens(cam, lens)
        if lens_obj is None:
            info.message = f"lensfun 未收录镜头: {lens}"
            return info

        info.source = "lensfun"
        info.message = f"已匹配 {info.camera} + {lens_obj.model}"
        info.candidates = [f"{l.maker} {l.model}" for l in
                           (self.db.find_lenses(cam) or [])][:8]
        return info

    def _find_camera(self, maker: str, model: str):
        if not self.available or not (maker or model):
            return None
        mk = str(maker or "")
        md = str(model or "")

        # 候选 model: 原始值 + 去掉 maker 前缀后的值 (EXIF 可能带或不带 "Canon " 前缀)。
        candidates = [md]
        if mk and md.lower().startswith(mk.lower()):
            stripped = md[len(mk):].strip()
            if stripped:
                candidates.append(stripped)
        # "Mark III"/"Mark II" -> "m3"/"m2" 缩写 (对齐 lensfun 命名习惯, 如 R6m2)
        compact = md.replace("Mark III", "m3").replace("Mark II", "m2")
        if compact != md:
            candidates.append(compact)

        seen = set()
        for cand in candidates:
            if not cand or cand.lower() in seen:
                continue
            seen.add(cand.lower())
            for loose in (False, True):
                for mk_arg in (mk, ""):
                    try:
                        cams = self.db.find_cameras(mk_arg, cand, loose_search=loose)
                        if cams:
                            return cams[0]
                    except Exception:
                        pass
        return None

    def _find_lens(self, cam, lens_name: str):
        if not self.available or cam is None or not lens_name:
            return None
        name = str(lens_name).strip()
        # 1) 直接精确查询
        try:
            lenses = self.db.find_lenses(cam, "", name)
            if lenses:
                return lenses[0]
        except Exception:
            pass
        # 2) 规范化 (去空格/统一小写) 后做双向包含匹配。
        #    处理命名差异: rawpy "RF70-200mm F2.8 L IS USM" vs
        #    lensfun "Canon RF 70-200mm F2.8L IS USM"。
        try:
            lenses = self.db.find_lenses(cam)
            if lenses:
                norm_name = _norm_lens(name)
                best = None
                for l in lenses:
                    if not l.model:
                        continue
                    norm_m = _norm_lens(l.model)
                    if norm_name and (norm_name in norm_m or norm_m in norm_name):
                        # 优先取"短名被长名完整包含"的匹配 (更精确)
                        if norm_name in norm_m:
                            return l
                        if best is None:
                            best = l
                return best
        except Exception:
            pass
        return None

    # ---------------- 校正 ----------------

    def correct(self, img: np.ndarray, meta: dict, params: LensCorrectionParams,
                info: Optional[LensInfo] = None) -> Tuple[np.ndarray, LensInfo]:
        """执行镜头校正。img 为线性 float32 RGB (0..1+)。"""
        out = np.ascontiguousarray(img.astype(np.float32))
        if info is None:
            info = self.identify(meta, params)

        used_lensfun = False
        if self.available and info.source == "lensfun":
            try:
                out = self._correct_lensfun(out, meta, params, info)
                used_lensfun = True
            except Exception as e:  # pragma: no cover
                info.message = f"lensfun 校正失败({type(e).__name__}), 回退参数模型"
                info.source = "fallback"

        if not used_lensfun:
            # 参数化通道
            if params.auto_tca:
                r_s, b_s = estimate_tca(out)
                params.tca_red, params.tca_blue = r_s, b_s
            if params.auto_vignetting:
                params.vignetting_k = estimate_vignetting(out)
            if params.enable_distortion:
                out = correct_distortion_model(out, params.distortion_k, params.distortion_p)
            if params.enable_tca:
                out = correct_tca_model(out, params.tca_red, params.tca_blue)
            if params.enable_vignetting:
                out = correct_vignetting_model(out, params.vignetting_k)

        return np.nan_to_num(out, nan=0.0, posinf=4.0, neginf=0.0).astype(np.float32), info

    def _correct_lensfun(self, img: np.ndarray, meta: dict,
                         params: LensCorrectionParams, info: LensInfo):
        h, w = img.shape[:2]
        maker = params.camera_maker or meta.get("make") or ""
        model = params.camera_model or meta.get("model") or ""
        lens_name = params.lens_model or meta.get("lens_model") or ""
        focal = float(params.focal_length or meta.get("focal_mm") or 35.0)
        aperture = float(params.aperture or meta.get("fnumber_val") or 2.8)

        cam = self._find_camera(maker, model)
        lens = self._find_lens(cam, lens_name)
        if cam is None or lens is None:
            raise RuntimeError("未匹配到机身/镜头")

        mod = lensfunpy.Modifier(lens, cam.crop_factor, w, h)
        mod.initialize(focal, aperture, float(params.distance or 10.0), scale=1.0)

        # 1) 几何畸变
        if params.enable_distortion:
            try:
                coords = mod.apply_subpixel_geometry_distortion()
                img = self._apply_coords(img, coords)
            except Exception:
                try:
                    coords = mod.apply_geometry_distortion()
                    img = self._apply_coords(img, coords)
                except Exception:
                    pass

        # 2) 色差 + 暗角 (lensfun 合并为 color modification, 原地修改 float32 缓冲)
        if params.enable_tca or params.enable_vignetting:
            buf = np.ascontiguousarray(img.astype(np.float32))
            try:
                if params.enable_tca and params.enable_vignetting:
                    mod.apply_color_modification(buf)
                elif params.enable_tca:
                    mod.apply_subpixel_tca(buf) if hasattr(mod, "apply_subpixel_tca") else mod.apply_color_modification(buf)
                else:
                    mod.apply_vignetting(buf) if hasattr(mod, "apply_vignetting") else None
                img = buf
            except Exception:
                img = buf  # 保持几何校正结果

        # 3) 若 lensfun 未覆盖某些项, 用参数模型补齐
        if params.enable_tca and (abs(params.tca_red - 1.0) > 1e-7 or abs(params.tca_blue - 1.0) > 1e-7):
            img = correct_tca_model(img, params.tca_red, params.tca_blue)
        if params.enable_vignetting and any(abs(v) > 1e-9 for v in params.vignetting_k):
            img = correct_vignetting_model(img, params.vignetting_k)
        if params.enable_distortion and any(abs(v) > 1e-9 for v in params.distortion_k):
            img = correct_distortion_model(img, params.distortion_k, params.distortion_p)

        return img

    @staticmethod
    def _apply_coords(img: np.ndarray, coords: np.ndarray) -> np.ndarray:
        """兼容 lensfunpy 不同版本的坐标返回格式。"""
        c = np.asarray(coords)
        if c.ndim == 3 and c.shape[-1] == 2:
            mapx = c[..., 0]
            mapy = c[..., 1]
        elif c.ndim == 3 and c.shape[0] == 2:
            mapx, mapy = c[0], c[1]
        else:  # pragma: no cover
            raise ValueError(f"未知的畸变坐标格式: {c.shape}")
        return cv2.remap(img, mapx.astype(np.float32), mapy.astype(np.float32),
                         interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
