"""诊断色彩白点: 确定 LibRaw 标定矩阵的实际白点, 检验是否需要色适应。"""
import sys
sys.path.insert(0, '.')
import numpy as np
import rawpy
from rawforge.core.decode import decode

SRC = 'samples/Nikon_-_D7200_-_12bit_.nef'

with rawpy.imread(SRC) as r:
    m = np.asarray(r.rgb_xyz_matrix, dtype=np.float64)[:3, :3]   # XYZ -> cam
    cam_wb = np.asarray(r.camera_whitebalance, dtype=np.float64)[:3]
    day_wb = np.asarray(r.daylight_whitebalance, dtype=np.float64)[:3]
    print("M (XYZ -> cam):\n", m)
    print("\ncamera_whitebalance  :", cam_wb, " 归一化:", cam_wb / cam_wb[1])
    print("daylight_whitebalance:", day_wb, " 归一化:", day_wb / day_wb[1])

cam_to_xyz = np.linalg.inv(m)


def mccamy(xy):
    x, y = xy
    n = (x - 0.3320) / (0.1858 - y)
    return 437 * n ** 3 + 3601 * n ** 2 + 6861 * n + 5517


def white_point_of(neutral_dir, label):
    xyz = cam_to_xyz @ neutral_dir
    xyz = xyz / xyz[1]
    s = xyz.sum()
    xy = (xyz[0] / s, xyz[1] / s)
    cct = mccamy(xy)
    print(f"\n{label}")
    print(f"   XYZ (Y=1)     : {xyz[0]:.4f}, {xyz[1]:.4f}, {xyz[2]:.4f}")
    print(f"   xy 色度       : {xy[0]:.4f}, {xy[1]:.4f}")
    print(f"   相关色温 CCT  : {cct:.0f} K")
    return xyz, cct


# 相机白平衡方向 (拍摄光源)
white_point_of(1.0 / (cam_wb / cam_wb[1]), "camera_wb 中性方向")
# 日光白平衡方向 (矩阵标定光源)
white_point_of(1.0 / (day_wb / day_wb[1]), "daylight_wb 中性方向 (标定光源)")

print("\n参考白点:")
for name, xy in (("D50", (0.34567, 0.35850)), ("D55", (0.33242, 0.34743)),
                 ("D60", (0.32168, 0.33767)), ("D65", (0.31271, 0.32902))):
    print(f"   {name}: xy=({xy[0]:.4f}, {xy[1]:.4f})  CCT={mccamy(xy):.0f} K")

# 检查: 若直接用 M 的行做 cam->XYZ (错误用法) 会怎样
print("\n=== 错误用法对照: 直接用 M 当 cam->XYZ ===")
wrong = m @ np.array([1.0, 1.0, 1.0])
print("   中性(1,1,1) ->", wrong / wrong[1])

# 检验 sRGB 目标白点: D65 经过我们的链路应映射回中性
print("\n=== 目标检验: XYZ(D65 白点) 经 XYZ->sRGB->相机空间 的往返 ===")
from rawforge.core import color as C
xyz_d65 = np.array([0.95047, 1.0, 1.08883])
lin = C.xyz_to_rgb(xyz_d65.astype(np.float32).reshape(1, 1, 3), "srgb").ravel()
print("   D65 -> linear sRGB :", lin, " (三者应相等)")
xyz_back = C.rgb_to_xyz(lin.astype(np.float32).reshape(1, 1, 3), "srgb").ravel()
print("   往返 -> XYZ        :", xyz_back)
print("   往返误差           :", np.abs(xyz_back - xyz_d65).max())
