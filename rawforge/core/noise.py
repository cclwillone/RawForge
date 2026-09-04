"""
噪声剖面标定 (Noise Profiling) —— 本项目的画质核心。

为什么必须做这一步:
    通用降噪器用"全局强度"参数, 暗部欠降噪、亮部过涂抹。
    真实传感器噪声服从泊松-高斯模型:

        var(I) = a * I + b

    其中 a 为散粒噪声(信号越强噪声越大), b 为读出噪声(与信号无关)。
    标定出 (a, b) 后, 可为每个像素算出它应有的 sigma, 实现"按噪声下药":
    暗部重降噪、亮部轻降噪、平坦区强、纹理区弱。

    这正是 darktable 的 "denoise (profiled)" 与 DxO DeepPRIME 的共同基础。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import numpy as np

__all__ = ["NoiseProfile", "estimate_noise_profile", "sigma_map", "estimate_sensor_read_noise", "profile_from_iso"]


@dataclass
class NoiseProfile:
    """泊松-高斯噪声模型参数。

    属性:
        a: 散粒噪声系数 (var = a * I)
        b: 读出噪声方差 (常数项)
        scale: 标定所用的数值尺度 (1.0 表示已归一化到 0..1 线性域)
        source: 标定来源 ("sensor" / "image" / "iso_estimate")
        confidence: 0~1, 拟合质量
    """

    a: float = 1e-4
    b: float = 1e-8
    scale: float = 1.0
    source: str = "unknown"
    confidence: float = 0.0

    def sigma(self, intensity) -> np.ndarray:
        """给定亮度(线性 0..1), 返回该处的噪声标准差。"""
        i = np.maximum(np.asarray(intensity, dtype=np.float32), 0.0)
        return np.sqrt(np.maximum(self.a * i + self.b, 0.0)).astype(np.float32)

    def sigma_median(self) -> float:
        """图像中位亮度处的典型 sigma, 用于参数自动推荐。"""
        return float(np.sqrt(max(self.a * 0.18 + self.b, 0.0)))

    def iso_equivalent(self) -> float:
        """粗略反推等效 ISO (仅用于 UI 显示)。"""
        s = self.sigma_median()
        if s <= 0:
            return 0.0
        return float(np.clip((s / 2.5e-4) ** 2 * 100.0, 25.0, 409600.0))

    def to_dict(self):
        return asdict(self)

    def __str__(self):
        return (f"NoiseProfile(a={self.a:.3e}, b={self.b:.3e}, "
                f"sigma@0.18={self.sigma_median():.5f}, source={self.source}, "
                f"conf={self.confidence:.2f})")


# --------------------------------------------------------------------------
# 从传感器光学遮罩区估计读出噪声
# --------------------------------------------------------------------------

def estimate_sensor_read_noise(sensor: np.ndarray, black_level: float = 0.0,
                               white_level: float = 65535.0,
                               sizes: Optional[dict] = None) -> Optional[float]:
    """利用 RAW 的光学遮罩区(optical black)估计读出噪声。

    sensor 为未减黑电平的拜耳原始阵列 (rawpy.raw_image)。
    遮罩区位于有效画面之外 (左侧 left 列、上方 top 行), 不接收光线,
    其起伏纯粹是读出噪声, 是最干净的噪声样本。
    """
    if sensor is None:
        return None
    try:
        arr = np.asarray(sensor)
        if arr.ndim != 2:
            return None
        h, w = arr.shape
        top = left = 0
        iw, ih = w, h
        if sizes:
            top = int(sizes.get("top", 0) or 0)
            left = int(sizes.get("left", 0) or 0)
            iw = int(sizes.get("iwidth", 0) or w)
            ih = int(sizes.get("iheight", 0) or h)

        candidates = []
        # 上方遮罩带
        if top >= 4:
            candidates.append(arr[max(0, top // 4):top, left:left + max(iw, 0)])
        # 左侧遮罩带
        if left >= 4:
            candidates.append(arr[top:top + max(ih, 0), max(0, left // 4):left])
        # 若没有遮罩区, 退化为取最外圈
        if not candidates:
            candidates.append(arr[: max(2, h // 200), :])
            candidates.append(arr[:, : max(2, w // 200)])

        sigmas = []
        for band in candidates:
            band = band[np.isfinite(band)]
            if band.size < 200:
                continue
            v = band.astype(np.float32)
            # 用 MAD 抗异常点
            med = np.median(v)
            mad = np.median(np.abs(v - med))
            if mad > 0:
                sigmas.append(float(1.4826 * mad))

        if not sigmas:
            return None
        read_noise_adu = float(np.median(sigmas))
        span = max(float(white_level) - float(black_level), 1.0)
        return read_noise_adu / span
    except Exception:
        return None


# --------------------------------------------------------------------------
# 从图像分块统计拟合噪声剖面
# --------------------------------------------------------------------------

def _noise_residual(gray: np.ndarray) -> np.ndarray:
    """提取噪声残差: 3x3 盒式高通。

    为什么必须先去趋势:
        直接在块内算标准差会把"渐变的坡度"也算成噪声 —— 天空的渐变、
        虚化的过渡都会让噪声被严重高估。3x3 盒式高通对任意线性渐变
        (局部一阶多项式) 的响应恒为 0, 只保留真正的随机起伏。

    增益校正:
        hp = n - (1/9)Σn  中, 中心项系数 8/9, 其余 8 项各 -1/9
        var(hp) = (64/81 + 8/81)·σ² = 8/9·σ²
        故 σ = std(hp) / sqrt(8/9)
    """
    try:
        import cv2
        blur = cv2.boxFilter(gray, -1, (3, 3), normalize=True,
                             borderType=cv2.BORDER_REPLICATE)
    except Exception:
        # 无 OpenCV 时的等价实现
        k = np.ones((3, 3), dtype=np.float32) / 9.0
        p = np.pad(gray, 1, mode="edge")
        blur = sum(shifted * w for shifted, w in zip(
            [p[i:i + gray.shape[0], j:j + gray.shape[1]]
             for i in range(3) for j in range(3)], k.ravel()))
    hp = (gray - blur).astype(np.float32)
    return hp * np.float32(1.0 / np.sqrt(8.0 / 9.0))


def _block_mean_mad(gray: np.ndarray, block: int = 16) -> Tuple[np.ndarray, np.ndarray]:
    """分块返回 (均值, 噪声标准差)。

    用高通残差代替原始块内标准差, 使统计对渐变、坡度免疫。
    """
    h, w = gray.shape
    bh, bw = h // block, w // block
    if bh < 2 or bw < 2:
        block = max(4, min(h, w) // 8)
        bh, bw = h // block, w // block
    if bh < 2 or bw < 2:
        return np.array([]), np.array([])

    hp = _noise_residual(gray)

    g = gray[: bh * block, : bw * block].reshape(bh, block, bw, block)
    means = g.mean(axis=(1, 3))

    # 残差的分块标准差: 用 MAD 抗离群点 (星点、坏点、强边缘残留)
    n = hp[: bh * block, : bw * block].reshape(bh, block, bw, block)
    mads = np.median(np.abs(n), axis=(1, 3))
    stds = 1.4826 * mads
    return means.ravel(), stds.ravel()


def estimate_noise_profile(
    linear: np.ndarray,
    block: int = 16,
    wb_gains: Optional[np.ndarray] = None,
    read_noise: Optional[float] = None,
) -> NoiseProfile:
    """从线性图像分块统计拟合泊松-高斯噪声模型。

    步骤:
        1. 转亮度, 分块求 (mean, std)
        2. 用局部梯度剔除纹理块 (纹理会被误判为噪声)
        3. 按亮度分箱取中位方差
        4. 最小二乘拟合 var = a * mean + b
    """
    img = np.asarray(linear, dtype=np.float32)
    if img.ndim == 3:
        if wb_gains is not None:
            g = np.asarray(wb_gains, dtype=np.float32).ravel()[:3]
            g = g / max(float(g.mean()), 1e-6)
            img = img * g.reshape(1, 1, 3)
        gray = (0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2])
    else:
        gray = img
    gray = np.nan_to_num(gray, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)

    # 纹理剔除: 用 3x3 拉普拉斯的能量判断平坦区
    try:
        import cv2
        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        flat = np.abs(lap)
    except Exception:
        flat = np.zeros_like(gray)

    means, stds = _block_mean_mad(gray, block)
    if means.size == 0:
        return NoiseProfile(source="image", confidence=0.0)

    # 与平均亮度对应的块级纹理度
    h, w = gray.shape
    bh, bw = h // max(block, 1), w // max(block, 1)
    if bh >= 2 and bw >= 2:
        f = flat[: bh * block, : bw * block].reshape(bh, block, bw, block)
        texture = np.median(f, axis=(1, 3)).ravel()
    else:
        texture = np.zeros_like(means)

    valid = (means > 1e-5) & (stds > 0) & (texture < np.percentile(texture, 70))
    if valid.sum() < 8:
        valid = (means > 1e-5) & (stds > 0)
    if valid.sum() < 8:
        return NoiseProfile(source="image", confidence=0.0)

    m, s = means[valid], stds[valid]
    var = s ** 2

    # 分箱取中位
    nbins = min(24, max(4, int(np.sqrt(m.size))))
    lo, hi = float(np.percentile(m, 1)), float(np.percentile(m, 99))
    if hi <= lo:
        hi = float(m.max()) if m.size else 1.0
        lo = float(m.min()) if m.size else 0.0
    if hi <= lo:
        return NoiseProfile(source="image", confidence=0.0)

    edges = np.linspace(lo, hi, nbins + 1)
    xs, ys = [], []
    for i in range(nbins):
        sel = (m >= edges[i]) & (m < edges[i + 1])
        if sel.sum() >= 3:
            xs.append(float(np.median(m[sel])))
            ys.append(float(np.median(var[sel])))
    if len(xs) < 3:
        xs, ys = list(m.astype(float)), list(var.astype(float))

    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)

    a = b = 0.0
    conf = 0.0
    try:
        # 加权最小二乘 (点数多的箱权重高)
        A = np.stack([xs, np.ones_like(xs)], axis=1)
        sol, *_ = np.linalg.lstsq(A, ys, rcond=None)
        a, b = float(sol[0]), float(sol[1])
        pred = A @ sol
        ss_res = float(np.sum((ys - pred) ** 2))
        ss_tot = float(np.sum((ys - ys.mean()) ** 2))
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        conf = float(np.clip(r2, 0.0, 1.0))
    except Exception:
        a, b = 1e-4, float(np.median(var)) if var.size else 1e-8

    a = float(np.clip(a, 1e-8, 0.5))
    b = float(np.clip(b, 1e-12, 0.5))

    # 若能用传感器遮罩区得到读出噪声, 用它替换常数项 (更可靠)
    if read_noise is not None and read_noise > 0:
        b = float(np.clip(read_noise ** 2, 1e-12, 0.5))
        source = "sensor"
        conf = max(conf, 0.85)
    else:
        source = "image"

    return NoiseProfile(a=a, b=b, scale=1.0, source=source, confidence=conf)


def sigma_map(intensity: np.ndarray, profile: NoiseProfile) -> np.ndarray:
    """生成逐像素 sigma 图。

    intensity 应先用平滑滤波去噪(否则噪声会让 sigma 被高估),
    典型做法: 传入原图的 box/gauss 模糊版本。
    """
    return profile.sigma(intensity)


def profile_from_iso(iso: float) -> NoiseProfile:
    """无 RAW 时的粗略估计: 依据典型 APS-C 传感器的经验曲线。"""
    try:
        iso = float(iso)
    except Exception:
        iso = 400.0
    iso = float(np.clip(iso, 25.0, 409600.0))
    # 经验: 中位亮度处 sigma 约与 ISO 的平方根成正比
    sigma_mid = 1.2e-4 * np.sqrt(iso)
    a = (sigma_mid ** 2) / 0.18
    b = max(a * 0.02, 1e-9)
    return NoiseProfile(a=float(a), b=float(b), source="iso_estimate", confidence=0.3)
