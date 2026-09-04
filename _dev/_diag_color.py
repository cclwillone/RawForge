"""验证色彩管线正确性: 与 rawpy/LibRaw 原生 sRGB 输出做数值对比。

如果我们的 "相机空间 -> XYZ -> sRGB" 链路正确, 那么在
关闭降噪/镜头校正、仅做相机白平衡 + 白点归一化后,
输出应与 LibRaw 自己的 output_color=sRGB 结果高度接近
(允许白点/曝光策略带来的整体亮度差异, 但色相必须一致)。
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import rawpy
from rawforge.core.decode import decode
from rawforge.core import color as C

SRC = 'samples/Nikon_-_D7200_-_12bit_.nef'

# ---------- 参考: LibRaw 原生 sRGB 输出 (线性域对比, 排除 gamma 差异) ----------
with rawpy.imread(SRC) as r:
    ref = r.postprocess(
        demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
        use_camera_wb=True,
        no_auto_bright=False,
        output_color=rawpy.ColorSpace.sRGB,
        output_bps=16,
        gamma=(1.0, 1.0),        # 线性输出, 排除 gamma 曲线差异
    )
ref = ref.astype(np.float32) / 65535.0

# ---------- 我们的链路 ----------
dec = decode(SRC, demosaic='ahd')
print("cam_xyz (camera -> XYZ):")
print(dec.cam_xyz)

# 中性灰验证: 用相机白平衡后的中性值应映射回接近 D50/D65 白点
wb = np.asarray(dec.camera_wb[:3], dtype=np.float64)
wb = wb / wb[1]
neutral_cam = (1.0 / wb)          # 相机空间中代表中性色的方向
xyz_neutral = dec.cam_xyz.astype(np.float64) @ neutral_cam
xyz_neutral = xyz_neutral / xyz_neutral[1]
print("\n中性色校验 (相机空间 1/wb 方向映射到 XYZ):")
print(f"  X/Y = {xyz_neutral[0]:.4f}   Z/Y = {xyz_neutral[2]:.4f}")
print(f"  D50 参考: X/Y = 0.9642  Z/Y = 0.8252")
print(f"  D65 参考: X/Y = 0.9505  Z/Y = 1.0890")

# 白平衡 + 色彩转换 (与 pipeline 一致: pre_mul = wb / daylight_wb)
gains = dec.camera_wb[:3].astype(np.float32).copy()
day_wb = dec.daylight_wb[:3].astype(np.float32).copy() if dec.daylight_wb is not None else np.ones(3, np.float32)
pre_mul = gains / day_wb
pre_mul = pre_mul / float(np.exp(np.mean(np.log(pre_mul))))
cam = dec.linear * pre_mul.reshape(1, 1, 3)
xyz = cam @ dec.cam_xyz.T
lin_srgb = C.xyz_to_rgb(xyz, "srgb")
lin_srgb = C.normalize_white(lin_srgb, 99.9)
ours = lin_srgb   # 线性域直接对比 (不做 gamma)

print("\n线性 sRGB 负像素比例: %.2f%%" % (100.0 * np.mean(lin_srgb < 0)))

# ---------- 对比 ----------
h = min(ours.shape[0], ref.shape[0])
w = min(ours.shape[1], ref.shape[1])
a = ours[:h, :w]
b = ref[:h, :w]

# 色相对比: 归一化掉整体亮度差 (比较色度坐标)
def chroma_coords(img):
    s = img.sum(axis=2) + 1e-6
    return img / s[..., None]

ca, cb = chroma_coords(a), chroma_coords(b)
# 只统计有内容的像素 (线性域, 亮度取对数域更稳)
valid = (b.max(axis=2) > 0.02) & (b.max(axis=2) < 1.0) & (a.max(axis=2) > 0.01)
if valid.sum() > 1000:
    d = np.abs(ca[valid] - cb[valid]).mean(axis=0)
    print(f"\n有效对比像素: {valid.sum():,}")
    print(f"  色度坐标平均绝对偏差 (R,G,B): {d[0]:.4f}, {d[1]:.4f}, {d[2]:.4f}")
    print(f"  整体色度偏差: {d.mean():.4f}  (<0.03 视为色彩一致)")
else:
    print("有效像素不足, 无法对比")

# 亮度相关
lum_a = a.mean(axis=2)
lum_b = b.mean(axis=2)
print(f"\n亮度  ours mean={lum_a.mean():.4f}   LibRaw mean={lum_b.mean():.4f}")
corr = np.corrcoef(lum_a.ravel(), lum_b.ravel())[0, 1]
print(f"亮度相关性: {corr:.4f}  (>0.95 表示结构一致)")

# 保存对比图
from rawforge.core import io_utils
import cv2
out = np.concatenate([cv2.cvtColor((a * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
                      cv2.cvtColor((b * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)], axis=1)
io_utils.imwrite("_diag_compare.png", out)
print("\n对比图已保存: _diag_compare.png  (左=我们, 右=LibRaw)")
