"""诊断: 验证块统计与噪声拟合的准确性 (修正版)。

注意: 三通道独立噪声经亮度加权 (0.2126,0.7152,0.0722) 后,
      亮度域 sigma = sigma_per_channel * sqrt(0.2126^2+0.7152^2+0.0722^2)
                   = sigma_per_channel * 0.7496
      因此用 sigma_ref = 0.7496 * sigma_channel 作为亮度域真值。
"""
import sys
sys.path.insert(0, '.')
import numpy as np
from rawforge.core.noise import _block_mean_mad, estimate_noise_profile

W = np.sqrt(0.2126 ** 2 + 0.7152 ** 2 + 0.0722 ** 2)  # 0.7496
rng = np.random.default_rng(0)
a_true, b_true = 2e-3, 4e-6

print("=" * 64)
print("测试 1: 恒定亮度平面 (检验块统计的绝对精度)")
print("=" * 64)
for level in (0.02, 0.18, 0.6):
    var_ch = a_true * level + b_true
    sig_ch = np.sqrt(var_ch)
    clean = np.full((512, 512, 3), level, dtype=np.float32)
    noisy = clean + rng.normal(0, sig_ch, clean.shape).astype(np.float32)
    gray = 0.2126 * noisy[..., 0] + 0.7152 * noisy[..., 1] + 0.0722 * noisy[..., 2]
    sig_ref = sig_ch * W
    means, stds = _block_mean_mad(gray.astype(np.float32), 16)
    est = float(np.median(stds))
    print(f"  level={level:<5} 亮度域sigma={sig_ref:.6f} 估计={est:.6f} "
          f"误差={100 * (est / sig_ref - 1):+.1f}%")

print()
print("=" * 64)
print("测试 2: 线性渐变 (检验对坡度的免疫 —— 修复前的重灾区)")
print("=" * 64)
h = w = 1024
yy, xx = np.mgrid[0:h, 0:w].astype(np.float32) / 1024.0
base = np.stack([xx, yy, (xx + yy) / 2], axis=-1).astype(np.float32)
lum = 0.2126 * base[..., 0] + 0.7152 * base[..., 1] + 0.0722 * base[..., 2]
sig_ch_map = np.sqrt(a_true * lum + b_true)
noise = rng.normal(0, 1, base.shape).astype(np.float32) * sig_ch_map[..., None]
noisy = base + noise
pf = estimate_noise_profile(noisy)
# 亮度域真值: var_lum = W^2 * var_ch = W^2 * (a*I + b)
a_ref, b_ref = W ** 2 * a_true, W ** 2 * b_true
sig_ref = np.sqrt(a_ref * 0.18 + b_ref)
print(f"  真实 (亮度域) a={a_ref:.3e} b={b_ref:.3e} sigma@0.18={sig_ref:.6f}")
print(f"  估计          a={pf.a:.3e} b={pf.b:.3e} sigma@0.18={pf.sigma_median():.6f} "
      f"误差={100 * (pf.sigma_median() / sig_ref - 1):+.1f}%  置信={pf.confidence:.2f}")

print()
print("=" * 64)
print("测试 3: 带纹理的真实感图像 (含正弦纹理 + 渐变 + 噪声)")
print("=" * 64)
tex = (0.5 + 0.5 * np.sin(xx * 60 * np.pi)) * (0.5 + 0.5 * np.sin(yy * 40 * np.pi))
base3 = np.stack([lum * 0.9 + tex * 0.05, lum * 1.0 + tex * 0.05, lum * 0.8 + tex * 0.05],
                 axis=-1).astype(np.float32)
lum3 = 0.2126 * base3[..., 0] + 0.7152 * base3[..., 1] + 0.0722 * base3[..., 2]
sig3 = np.sqrt(a_true * lum3 + b_true)
noisy3 = base3 + rng.normal(0, 1, base3.shape).astype(np.float32) * sig3[..., None]
pf3 = estimate_noise_profile(noisy3)
sig_ref3 = np.sqrt(a_ref * 0.18 + b_ref)
print(f"  真实 sigma@0.18={sig_ref3:.6f}")
print(f"  估计 sigma@0.18={pf3.sigma_median():.6f} 误差={100 * (pf3.sigma_median() / sig_ref3 - 1):+.1f}% "
      f"置信={pf3.confidence:.2f}  a={pf3.a:.3e} b={pf3.b:.3e}")

print()
print("=" * 64)
print("测试 4: 无噪声干净图 (不应产生虚假噪声估计)")
print("=" * 64)
pf4 = estimate_noise_profile(base.astype(np.float32))
print(f"  干净图估计 sigma@0.18={pf4.sigma_median():.6f} (应接近 0) 置信={pf4.confidence:.2f}")
