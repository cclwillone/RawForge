"""
降噪引擎: 噪声剖面自适应。

算法组成 (全部自研, 基于 numpy/OpenCV 向量化实现):
    1. NLM (Non-Local Means, 非局部均值)
       以图像块为相似度单位, 在搜索窗内寻找结构相似的区域做加权平均。
       相比局部滤波, 它能在抹平噪声的同时保住纹理与细线。
       本实现要点:
       - 用盒式滤波(boxFilter)在 O(1) 内算出全图**块级**距离, 完全向量化
       - 权重由亮度通道统一计算后共享给 RGB, 避免通道间颜色渗透
       - 距离做 2*sigma^2 无偏修正, 防止高噪声下权重退化
       - 可选在低分辨率上算权重再上采样, 速度提升约 4 倍
    2. WT (Haar 小波软阈值 + 循环平移)
       循环平移消除小波的正交伪影(Gibbs), 对平坦区噪声极有效, 复杂度 O(N)。
    3. 亮度/色度分离
       人眼对亮度细节敏感、对色度噪声容忍度低 —— 亮度轻降噪保细节,
       色度重降噪去彩噪。这是 DeepPRIME 一类算法的基本策略。

为什么必须在线性域降噪:
    线性域噪声近似高斯且与信号强度可建模 (var = a*I + b);
    gamma 编码会压缩亮部、拉伸暗部, 使噪声变得非平稳, 模型失效。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import cv2
import numpy as np

__all__ = [
    "DenoiseParams", "denoise", "nlm_denoise", "wavelet_denoise",
    "bilateral_denoise", "median_denoise", "split_luma_chroma",
    "auto_params_from_profile",
]

SQRT2 = float(np.sqrt(2.0))


@dataclass
class DenoiseParams:
    """降噪参数。"""

    enable: bool = True
    method: str = "hybrid"        # hybrid / nlm / wavelet / bilateral / median
    strength: float = 1.0         # 总强度倍率 (0=关闭, 1=标准, 2=激进)
    luminance: float = 1.0        # 亮度降噪强度
    chroma: float = 0.6           # 色度降噪强度 (0..3, 默认保守以保护色彩层次)

    # NLM
    patch_radius: int = 1         # 相似度块半径 -> 块尺寸 2r+1
    search_radius: int = 4        # 搜索窗半径 -> (2r+1)^2 个候选
    nlm_h: float = 0.9            # 滤波系数 (越大越强, 0.6~1.4 合理)
    nlm_fast: bool = True         # 在低分辨率上算权重 (推荐, 视觉差异极小)

    # 小波
    wavelet_levels: int = 3
    wavelet_threshold: float = 2.4
    wavelet_cycle_spin: bool = True

    # 双边
    bilateral_d: int = 0
    bilateral_sigma_color: float = 0.08
    bilateral_sigma_space: float = 2.0

    # 细节保护
    detail_protect: float = 0.35  # 0=无保护, 1=完全保护纹理区
    sharpen_after: float = 0.0    # 降噪后锐化量 (0~1)

    def to_dict(self):
        return asdict(self)


# --------------------------------------------------------------------------
# 工具
# --------------------------------------------------------------------------

def _to_float32(img):
    return np.ascontiguousarray(np.asarray(img, dtype=np.float32))


def _shift(img, dx, dy):
    """按整数偏移平移图像。

    用 np.roll 实现 (C 层, 远快于切片复制)。边缘为环绕填充,
    对 NLM 而言边缘像素权重极低, 环绕带来的误差可忽略。
    """
    return np.roll(np.roll(img, dy, axis=0), dx, axis=1)


def split_luma_chroma(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """线性 RGB -> (亮度 Y, 色度残差)。色度用 R-Y, B-Y 两通道表示。

    用 Y = 0.2126R + 0.7152G + 0.0722B (Rec.709 亮度), 在**线性域**计算,
    保证亮度/色度的能量分离符合人眼感知权重。
    """
    y = (0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]).astype(np.float32)
    chroma = np.stack([img[..., 0] - y, img[..., 2] - y], axis=-1).astype(np.float32)
    return y, chroma


def _merge_luma_chroma(y: np.ndarray, chroma: np.ndarray) -> np.ndarray:
    r = y + chroma[..., 0]
    b = y + chroma[..., 1]
    g = (y - 0.2126 * r - 0.0722 * b) / 0.7152
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def _sigma_tiles(sigma_map: Optional[np.ndarray], shape, tile=64):
    """把 sigma map 降采样到 tile 网格再上采样回原尺寸, 减少逐像素开销。"""
    h, w = shape[:2]
    if sigma_map is None:
        return None
    s = np.asarray(sigma_map, dtype=np.float32)
    if s.shape != (h, w):
        s = cv2.resize(s, (w, h), interpolation=cv2.INTER_LINEAR)
    th, tw = max(1, h // tile), max(1, w // tile)
    if th < 1 or tw < 1:
        return s
    small = cv2.resize(s, (tw, th), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)


# --------------------------------------------------------------------------
# 1. NLM
# --------------------------------------------------------------------------

def nlm_denoise(
    img: np.ndarray,
    sigma: float = 0.01,
    sigma_map: Optional[np.ndarray] = None,
    params: Optional[DenoiseParams] = None,
    guide: Optional[np.ndarray] = None,
) -> np.ndarray:
    """非局部均值降噪 (向量化块匹配实现)。

    参数:
        img:       线性 float32 RGB
        sigma:     全局噪声标准差 (无 sigma_map 时使用)
        sigma_map: 逐像素噪声标准差 (来自噪声剖面), 优先使用
        params:    DenoiseParams
        guide:     用于计算相似度的引导图 (如预降噪结果), 提升权重鲁棒性
    """
    p = params or DenoiseParams()
    img32 = _to_float32(img)
    h, w = img32.shape[:2]

    sig = _sigma_tiles(sigma_map, img32.shape) if sigma_map is not None else None
    if sig is None:
        sig = np.full((h, w), float(max(sigma, 1e-6)), dtype=np.float32)

    # 权重计算分辨率
    scale = 2 if (p.nlm_fast and min(h, w) > 900) else 1
    if scale > 1:
        gh, gw = h // scale, w // scale
        base = cv2.resize(img32, (gw, gh), interpolation=cv2.INTER_AREA)
        if guide is not None:
            base = cv2.resize(_to_float32(guide), (gw, gh), interpolation=cv2.INTER_AREA)
        sig_s = cv2.resize(sig, (gw, gh), interpolation=cv2.INTER_AREA)
    else:
        base = _to_float32(guide) if guide is not None else img32
        sig_s = sig

    # 相似度基于亮度
    lum = (0.2126 * base[..., 0] + 0.7152 * base[..., 1] + 0.0722 * base[..., 2]).astype(np.float32)
    patch = 2 * int(max(p.patch_radius, 1)) + 1
    area = float(patch * patch)
    R = int(max(p.search_radius, 1))

    h2 = float(max(p.nlm_h, 1e-3)) ** 2

    # 各候选偏移
    offsets = [(dx, dy) for dy in range(-R, R + 1) for dx in range(-R, R + 1)
               if not (dx == 0 and dy == 0)]

    # 需要降噪的 RGB 工作图 (scale>1 时在低分辨率上累加再上采样)
    if scale > 1:
        gh_, gw_ = lum.shape
        img_work = cv2.resize(img32, (gw_, gh_), interpolation=cv2.INTER_AREA)
    else:
        img_work = img32

    acc_l = np.zeros(lum.shape, dtype=np.float32)
    wsum_l = np.ones(lum.shape, dtype=np.float32)
    acc_c = np.zeros_like(img_work)
    wsum_c = np.ones(img_work.shape[:2], dtype=np.float32)

    # 单个循环: 一次计算权重, 同时累加亮度与 RGB (避免重复 boxFilter)
    for dx, dy in offsets:
        shifted_lum = _shift(lum, dx, dy)
        diff2 = (lum - shifted_lum) ** 2
        dist = cv2.boxFilter(diff2, -1, (patch, patch), normalize=True,
                             borderType=cv2.BORDER_REPLICATE) * area
        # 无偏修正: 含噪块距离的期望含 2*sigma^2*area
        unbiased = np.maximum(dist - 2.0 * (sig_s ** 2) * area, 0.0)
        denom = (sig_s ** 2) * area * h2 + 1e-12
        weight = np.exp(-unbiased / denom).astype(np.float32)

        acc_l += weight * shifted_lum
        wsum_l += weight
        acc_c += weight[..., None] * _shift(img_work, dx, dy)
        wsum_c += weight

    lum_dn = acc_l / np.maximum(wsum_l, 1e-6)
    out_work = acc_c / np.maximum(wsum_c, 1e-6)[..., None]

    if scale > 1:
        out = cv2.resize(out_work, (w, h), interpolation=cv2.INTER_LINEAR)
        lum_dn = cv2.resize(lum_dn, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        out = out_work

    # 与引导亮度结果做通道一致性校正 (抑制颜色渗透)
    y_out = (0.2126 * out[..., 0] + 0.7152 * out[..., 1] + 0.0722 * out[..., 2])
    delta = (lum_dn - y_out).astype(np.float32)
    out = np.stack([out[..., 0] + delta, out[..., 1] + delta, out[..., 2] + delta], axis=-1)

    return np.clip(out, 0.0, None).astype(np.float32)


# --------------------------------------------------------------------------
# 2. 小波 (Haar + 循环平移软阈值)
# --------------------------------------------------------------------------

def _haar_step2d(x: np.ndarray):
    """一级 2D Haar 分解, 返回 (LL, LH, HL, HH)。要求偶数尺寸。"""
    h, w = x.shape[:2]
    if h % 2 or w % 2:
        x = x[: h - h % 2, : w - w % 2]
        h, w = x.shape[:2]
    a = (x[0::2, ...] + x[1::2, ...]) / SQRT2
    d = (x[0::2, ...] - x[1::2, ...]) / SQRT2
    LL = (a[:, 0::2, ...] + a[:, 1::2, ...]) / SQRT2
    LH = (a[:, 0::2, ...] - a[:, 1::2, ...]) / SQRT2
    HL = (d[:, 0::2, ...] + d[:, 1::2, ...]) / SQRT2
    HH = (d[:, 0::2, ...] - d[:, 1::2, ...]) / SQRT2
    return LL, LH, HL, HH


def _haar_inv2d(LL, LH, HL, HH):
    ah, aw = LL.shape[0], LL.shape[1]
    a = np.zeros((ah, aw * 2), dtype=LL.dtype)
    d = np.zeros((ah, aw * 2), dtype=LL.dtype)
    a[:, 0::2] = (LL + LH) / SQRT2
    a[:, 1::2] = (LL - LH) / SQRT2
    d[:, 0::2] = (HL + HH) / SQRT2
    d[:, 1::2] = (HL - HH) / SQRT2
    out = np.zeros((ah * 2, aw * 2), dtype=LL.dtype)
    out[0::2, :] = (a + d) / SQRT2
    out[1::2, :] = (a - d) / SQRT2
    return out


def _soft_threshold(x, t):
    return np.sign(x) * np.maximum(np.abs(x) - t, 0.0)


def wavelet_denoise(
    img: np.ndarray,
    sigma: float = 0.01,
    sigma_map: Optional[np.ndarray] = None,
    params: Optional[DenoiseParams] = None,
) -> np.ndarray:
    """Haar 小波软阈值降噪 (可选循环平移)。

    细节子带的噪声标准差与原始 sigma 相同 (Haar 是正交归一基),
    因此可直接用统一阈值; 逐像素 sigma 时按子带降采样使用。
    """
    p = params or DenoiseParams()
    img32 = _to_float32(img)
    h, w = img32.shape[:2]

    sig = _sigma_tiles(sigma_map, (h, w)) if sigma_map is not None else None
    if sig is None:
        sig = np.full((h, w), float(max(sigma, 1e-6)), dtype=np.float32)

    levels = int(np.clip(p.wavelet_levels, 1, 6))
    k = float(max(p.wavelet_threshold, 0.1))

    shifts = [(0, 0), (0, 1), (1, 0), (1, 1)] if p.wavelet_cycle_spin else [(0, 0)]
    accum = np.zeros((h, w, img32.shape[2]), dtype=np.float32)

    for sy, sx in shifts:
        cur = np.roll(np.roll(img32, sy, axis=0), sx, axis=1)
        s_cur = np.roll(np.roll(sig, sy, axis=0), sx, axis=1)
        chans = []
        for c in range(cur.shape[2]):
            chans.append(_wavelet_single(cur[..., c], s_cur, levels, k))
        res = np.stack(chans, axis=-1)
        accum += np.roll(np.roll(res, -sy, axis=0), -sx, axis=1)

    out = accum / len(shifts)
    return np.clip(out, 0.0, None).astype(np.float32)


def _wavelet_single(x: np.ndarray, sig: np.ndarray, levels: int, k: float) -> np.ndarray:
    """单通道小波降噪, 支持任意尺寸 (自动填充到 2^levels 的倍数)。"""
    h, w = x.shape
    pad_h = (2 ** levels - h % (2 ** levels)) % (2 ** levels)
    pad_w = (2 ** levels - w % (2 ** levels)) % (2 ** levels)
    if pad_h or pad_w:
        x = np.pad(x, ((0, pad_h), (0, pad_w)), mode="edge")
        sig = np.pad(sig, ((0, pad_h), (0, pad_w)), mode="edge")

    coeffs = []
    cur = x.astype(np.float32)
    cur_sig = sig.astype(np.float32)
    sigs = []
    for lv in range(levels):
        LL, LH, HL, HH = _haar_step2d(cur)
        sLL, sLH, sHL, sHH = _haar_step2d(cur_sig)
        coeffs.append((LH, HL, HH))
        # 子带噪声: 细节系数噪声 std = sigma (正交变换保能量), 阈值直接乘 k
        sigs.append((sLH, sHL, sHH))
        cur, cur_sig = LL, sLL

    # 阈值化
    for i in range(levels - 1, -1, -1):
        LH, HL, HH = coeffs[i]
        sLH, sHL, sHH = sigs[i]
        LH_t = _soft_threshold(LH, k * sLH)
        HL_t = _soft_threshold(HL, k * sHL)
        HH_t = _soft_threshold(HH, k * sHH)
        cur = _haar_inv2d(cur, LH_t, HL_t, HH_t)

    return cur[:h, :w].astype(np.float32)


# --------------------------------------------------------------------------
# 3. 其它降噪器
# --------------------------------------------------------------------------

def bilateral_denoise(img: np.ndarray, params: Optional[DenoiseParams] = None,
                      sigma: float = 0.01) -> np.ndarray:
    """双边滤波: 保边去噪, 对轻微噪声快速有效。"""
    p = params or DenoiseParams()
    img32 = _to_float32(img)
    sc = float(max(p.bilateral_sigma_color, 1e-4)) * (sigma / 0.01)
    sc = float(np.clip(sc, 1e-4, 1.0))
    out = cv2.bilateralFilter(img32, int(p.bilateral_d) or 5, sc,
                              float(max(p.bilateral_sigma_space, 0.5)))
    return np.clip(out, 0.0, None).astype(np.float32)


def median_denoise(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """中值滤波: 专门压制椒盐/坏点, 对高斯噪声效果有限。"""
    k = int(max(3, ksize // 2 * 2 + 1))
    out = cv2.medianBlur(_to_float32(img), k)
    return out.astype(np.float32)


# --------------------------------------------------------------------------
# 4. 细节保护与锐化
# --------------------------------------------------------------------------

def _detail_mask(img: np.ndarray, sigma: float) -> np.ndarray:
    """纹理保护掩膜: 平坦区=1(可强降噪), 纹理区=0(保留细节)。"""
    lum = img.mean(axis=2) if img.ndim == 3 else img
    lap = cv2.Laplacian(lum.astype(np.float32), cv2.CV_32F, ksize=3)
    energy = cv2.GaussianBlur(np.abs(lap), (0, 0), 2.0)
    # 噪声本身的拉普拉斯能量约为 sigma * 4
    noise_level = max(sigma * 4.0, 1e-6)
    ratio = energy / noise_level
    mask = 1.0 / (1.0 + np.maximum(ratio - 1.0, 0.0))
    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def unsharp_mask(img: np.ndarray, amount: float = 0.5, radius: float = 1.2) -> np.ndarray:
    """非锐化掩膜 (在高斯模糊残差上叠加细节)。"""
    if amount <= 0:
        return img
    blur = cv2.GaussianBlur(img, (0, 0), float(max(radius, 0.1)))
    high = img - blur
    return (img + float(amount) * high).astype(np.float32)


# --------------------------------------------------------------------------
# 5. 统一入口
# --------------------------------------------------------------------------

def auto_params_from_profile(profile) -> DenoiseParams:
    """依据噪声剖面自动推荐参数 (ISO 越高越强)。"""
    s = profile.sigma_median()
    strength = float(np.clip(0.55 + 3.2 * s, 0.3, 3.0))
    return DenoiseParams(
        strength=strength,
        luminance=strength,
        chroma=strength * 1.35,
        nlm_h=float(np.clip(0.75 + 1.2 * s, 0.6, 1.35)),
        search_radius=5 if s > 0.02 else 4,
        wavelet_threshold=float(np.clip(2.0 + 6.0 * s, 2.0, 3.2)),
    )


def denoise(
    img: np.ndarray,
    profile=None,
    params: Optional[DenoiseParams] = None,
    sigma_map: Optional[np.ndarray] = None,
) -> np.ndarray:
    """降噪主入口。

    参数:
        img:       线性 float32 RGB (0..1+)
        profile:   NoiseProfile, 用于自适应强度
        params:    DenoiseParams
        sigma_map: 可选的逐像素噪声图
    """
    p = params or DenoiseParams()
    if not p.enable:
        return img

    img32 = _to_float32(img)
    from .noise import NoiseProfile
    if profile is None:
        profile = NoiseProfile()
    if sigma_map is None:
        sigma_map = profile.sigma(np.clip(img32.mean(axis=2), 0.0, None))

    sigma_med = float(max(profile.sigma_median(), 1e-6))
    strength = float(np.clip(p.strength, 0.0, 3.0))

    # 保护掩膜: 纹理区减弱降噪
    if p.detail_protect > 0:
        mask = _detail_mask(img32, sigma_med)
        protect = 1.0 - float(np.clip(p.detail_protect, 0.0, 1.0)) * (1.0 - mask)
    else:
        protect = np.ones(img32.shape[:2], dtype=np.float32)

    method = (p.method or "hybrid").lower()

    if method == "nlm":
        out = nlm_denoise(img32, sigma_med, sigma_map, p)
    elif method == "wavelet":
        sp = DenoiseParams(**{**p.to_dict(), "wavelet_threshold": p.wavelet_threshold})
        out = wavelet_denoise(img32, sigma_med, sigma_map, sp)
    elif method == "bilateral":
        out = bilateral_denoise(img32, p, sigma_med)
    elif method == "median":
        out = median_denoise(img32, 3)
    else:
        # hybrid: 小波预降噪 -> NLM 主降噪 -> 色度强化
        pre = wavelet_denoise(img32, sigma_med, sigma_map,
                              DenoiseParams(**{**p.to_dict(),
                                               "wavelet_threshold": p.wavelet_threshold * 0.75}))
        out = nlm_denoise(img32, sigma_med, sigma_map, p, guide=pre)

    # 强度混合 (strength 控制降噪比例)
    if strength < 0.999:
        out = img32 + (out - img32) * strength
    elif strength > 1.001:
        out = img32 + (out - img32) * min(strength, 2.5)

    # 色度降噪: 亮度/色度分离后对色度追加一次强降噪
    # 色度噪点 (去马赛克后的彩斑) 是低频的, 大半径高斯即可强力压制,
    # 无需保边 (色度本就应平滑), 且避免对 2 通道做 bilateralFilter 的限制。
    chroma_strength = float(np.clip(p.chroma, 0.0, 3.0))
    if chroma_strength > 0.01:
        y, ch = split_luma_chroma(out)
        # 公式从 1.0+cs*1.6 改为 0.6+cs*0.8: 默认 cs=0.6 时 sigma=1.08 像素,
        # 足以压彩噪而不破坏色彩层次; 旧默认 1.4 时 sigma=3.24 会把花瓣纹理
        # 的细微色阶抹平, 在大尺寸图像上产生可察觉的色块伪影。
        sigma_ch = float(0.6 + chroma_strength * 0.8)
        ch_dn = cv2.GaussianBlur(ch, (0, 0), sigma_ch)
        out = _merge_luma_chroma(y, ch_dn.astype(np.float32))

    # 细节保护: 把原图的高频按保护强度加回
    if p.detail_protect > 0:
        blurred = cv2.GaussianBlur(out, (0, 0), 1.0)
        orig_hf = img32 - cv2.GaussianBlur(img32, (0, 0), 1.0)
        out = blurred + orig_hf * (1.0 - protect)[..., None]

    if p.sharpen_after > 0:
        out = unsharp_mask(out, float(p.sharpen_after) * 0.8, 1.1)

    return np.clip(np.nan_to_num(out, nan=0.0, posinf=4.0, neginf=0.0), 0.0, None).astype(np.float32)
