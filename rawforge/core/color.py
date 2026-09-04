"""
色彩科学与工程工具。

设计原则:
    RAW 管线必须在线性光(linear light)下运算, 最后一步才做 gamma 编码。
    所有函数输入输出均为 float32, 范围约定:
        linear  : 0.0 ~ 1.0 (可 >1 表示高光溢出)
        encoded : 0.0 ~ 1.0 (gamma 编码后)
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "SRGB_TO_XYZ", "XYZ_TO_SRGB", "ADOBE_TO_XYZ", "XYZ_TO_ADOBE",
    "PROPHOTO_TO_XYZ", "XYZ_TO_PROPHOTO", "REC2020_TO_XYZ", "XYZ_TO_REC2020",
    "MATRICES", "GAMMA_FUNCS", "INV_GAMMA_FUNCS",
    "srgb_to_linear", "linear_to_srgb", "apply_transfer", "apply_inverse_transfer",
    "rgb_to_xyz", "xyz_to_rgb", "rgb_to_rgb", "apply_matrix",
    "camera_to_xyz", "xyz_to_output", "apply_wb", "normalize_white",
    "kelvin_to_rgb_multipliers", "gray_world_wb", "safe_div",
]


# --------------------------------------------------------------------------
# 色彩空间矩阵 (D65 白点, 列主序展开为行矩阵, 用于 img @ M.T)
# --------------------------------------------------------------------------

SRGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=np.float32)

XYZ_TO_SRGB = np.array([
    [3.2404542, -1.5371385, -0.4985314],
    [-0.9692660, 1.8760108, 0.0415560],
    [0.0556434, -0.2040259, 1.0572252],
], dtype=np.float32)

ADOBE_TO_XYZ = np.array([
    [0.5767309, 0.1855540, 0.1881852],
    [0.2973769, 0.6273491, 0.0752741],
    [0.0270343, 0.0706872, 0.9911085],
], dtype=np.float32)

XYZ_TO_ADOBE = np.array([
    [2.0413690, -0.5649464, -0.3446944],
    [-0.9692660, 1.8760108, 0.0415560],
    [0.0134474, -0.1183897, 1.0154096],
], dtype=np.float32)

PROPHOTO_TO_XYZ = np.array([
    [0.7977604896723027, 0.13518583717574031, 0.0313493495815248],
    [0.2880711282292934, 0.7118432178101014, 0.00008565396060525902],
    [0.0, 0.0, 0.8251046025104602],
], dtype=np.float32)

XYZ_TO_PROPHOTO = np.array([
    [1.3457989731028281, -0.25558010007997534, -0.05110628506753401],
    [-0.5446224939028347, 1.5082327413132781, 0.02053603239147973],
    [0.0, 0.0, 1.2119675456389454],
], dtype=np.float32)

REC2020_TO_XYZ = np.array([
    [0.6369580483012914, 0.14461690358620832, 0.1688809751641721],
    [0.2627002120112671, 0.6779980715188708, 0.05930171646986196],
    [0.0, 0.028072693049087428, 1.060985057710791],
], dtype=np.float32)

XYZ_TO_REC2020 = np.array([
    [1.7166511879712674, -0.35567078377639233, -0.25336628137365974],
    [-0.6666843518324892, 1.6164812366349395, 0.01576854581391113],
    [0.017639857445310783, -0.042770613257808524, 0.9421031212354738],
], dtype=np.float32)

# 名称 -> (to_xyz, from_xyz, 传输函数标识)
MATRICES = {
    "srgb": (SRGB_TO_XYZ, XYZ_TO_SRGB, "srgb"),
    "adobe_rgb": (ADOBE_TO_XYZ, XYZ_TO_ADOBE, "gamma_2.2"),
    "prophoto": (PROPHOTO_TO_XYZ, XYZ_TO_PROPHOTO, "gamma_1.8"),
    "rec2020": (REC2020_TO_XYZ, XYZ_TO_REC2020, "rec709"),
    "linear_raw": (None, None, "linear"),
}


# --------------------------------------------------------------------------
# 传输函数 (gamma)
# --------------------------------------------------------------------------

def srgb_to_linear(x):
    """sRGB 电光转换函数 (EOTF), 输出线性光。支持数组/LUT。"""
    x = np.asarray(x, dtype=np.float32)
    out = np.where(
        x <= 0.04045,
        x / 12.92,
        np.power(np.maximum((x + 0.055) / 1.055, 1e-8), 2.4),
    )
    return out.astype(np.float32)


def linear_to_srgb(x):
    """线性光 -> sRGB 编码 (OETF), 已做 0~1 裁剪。"""
    x = np.asarray(x, dtype=np.float32)
    xc = np.clip(x, 0.0, 1.0)
    out = np.where(
        xc <= 0.0031308,
        xc * 12.92,
        1.055 * np.power(np.maximum(xc, 1e-8), 1.0 / 2.4) - 0.055,
    )
    return out.astype(np.float32)


def _gamma_pure(g):
    def enc(x):
        return np.power(np.clip(x, 0.0, 1.0), 1.0 / g)

    def dec(x):
        return np.power(np.clip(x, 0.0, 1.0), g)

    return enc, dec


_G22_ENC, _G22_DEC = _gamma_pure(2.2)
_G18_ENC, _G18_DEC = _gamma_pure(1.8)


def _rec709_enc(x):
    xc = np.clip(x, 0.0, 1.0)
    return np.where(xc < 0.018, xc * 4.5, 1.099 * np.power(np.maximum(xc, 1e-8), 0.45) - 0.099)


def _rec709_dec(x):
    xc = np.clip(x, 0.0, 1.0)
    return np.where(xc < 0.081, xc / 4.5, np.power(np.maximum((xc + 0.099) / 1.099, 1e-8), 1.0 / 0.45))


def _identity(x):
    return np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)


GAMMA_FUNCS = {
    "srgb": linear_to_srgb,
    "gamma_2.2": lambda x: _G22_ENC(x).astype(np.float32),
    "gamma_1.8": lambda x: _G18_ENC(x).astype(np.float32),
    "rec709": lambda x: _rec709_enc(x).astype(np.float32),
    "linear": _identity,
}

INV_GAMMA_FUNCS = {
    "srgb": srgb_to_linear,
    "gamma_2.2": lambda x: _G22_DEC(x).astype(np.float32),
    "gamma_1.8": lambda x: _G18_DEC(x).astype(np.float32),
    "rec709": lambda x: _rec709_dec(x).astype(np.float32),
    "linear": _identity,
}


def apply_transfer(linear, space="srgb"):
    """线性光 -> 指定色彩空间的编码值。"""
    return GAMMA_FUNCS.get(space, linear_to_srgb)(linear)


def apply_inverse_transfer(encoded, space="srgb"):
    """编码值 -> 线性光。"""
    return INV_GAMMA_FUNCS.get(space, srgb_to_linear)(encoded)


# --------------------------------------------------------------------------
# 矩阵运算
# --------------------------------------------------------------------------

def apply_matrix(img, m):
    """对 HxWx3 图像右乘 3x3 矩阵: out = img @ M.T"""
    m = np.asarray(m, dtype=np.float32)
    return (img @ m.T).astype(np.float32)


def rgb_to_xyz(img, space="srgb"):
    m = MATRICES.get(space, MATRICES["srgb"])[0]
    if m is None:
        return img
    return apply_matrix(img, m)


def xyz_to_rgb(img, space="srgb"):
    m = MATRICES.get(space, MATRICES["srgb"])[1]
    if m is None:
        return img
    return apply_matrix(img, m)


def rgb_to_rgb(img, src="srgb", dst="srgb"):
    """在两个 RGB 空间之间做 Bradford 无关的直连转换 (经 XYZ 中转)。"""
    if src == dst:
        return img
    xyz = rgb_to_xyz(img, src)
    return xyz_to_rgb(xyz, dst)


def safe_div(a, b, eps=1e-8):
    return a / (np.asarray(b, dtype=np.float32) + eps)


def camera_to_xyz(cam_rgb, cam_xyz):
    """相机原生 RGB -> XYZ。cam_xyz 为 rawpy 的 cam_xyz 或 xyz_matrix (3x3/4x3)。"""
    m = np.asarray(cam_xyz, dtype=np.float64)
    if m.shape[0] >= 3 and m.shape[1] >= 3:
        m3 = m[:3, :3]
    else:
        return cam_rgb
    # rawpy 的 rgb_xyz_matrix 行和为 1, 需要按行和归一化后再用
    rowsum = m3.sum(axis=1)
    if np.all(rowsum > 0):
        m3 = m3 / rowsum[:, None]
    return (cam_rgb.astype(np.float32) @ np.linalg.inv(m3).T.astype(np.float32)).astype(np.float32)


def xyz_to_output(xyz, space="srgb"):
    """XYZ(D65) -> 目标 RGB 线性值。"""
    return xyz_to_rgb(xyz, space)


# --------------------------------------------------------------------------
# 白平衡
# --------------------------------------------------------------------------

def apply_wb(img, gains):
    """按通道增益做白平衡。gains 长度 3 (RGB) 或 4 (RGBG)。"""
    g = np.asarray(gains, dtype=np.float32).ravel()
    if g.size >= 4:
        g = g[:3]
    g = g / max(float(g.mean()), 1e-6) * float(np.mean(g[:3])) if False else g
    # 归一化: 保持整体亮度不变 (以最大增益为基准会变暗, 这里用几何平均)
    g = g / max(float(np.exp(np.mean(np.log(np.maximum(g, 1e-6))))), 1e-6)
    out = img * g.reshape(1, 1, 3)
    return out.astype(np.float32)


def normalize_white(img, percentile=99.9):
    """自动白点归一化: 让高百分位落在 1.0。"""
    hi = np.percentile(img, percentile)
    return (img / max(float(hi), 1e-6)).astype(np.float32)


def kelvin_to_rgb_multipliers(kelvin):
    """色温(K) -> 近似的 RGB 白平衡乘数 (Tanner Helland 近似, 再归一化)。"""
    k = float(np.clip(kelvin, 1000.0, 40000.0)) / 100.0
    if k <= 66.0:
        r = 255.0
        g = 99.4708025861 * np.log(k) - 161.1195681661
    else:
        r = 329.698727446 * np.power(k - 60.0, -0.1332047592)
        g = 288.1221695283 * np.power(k - 60.0, -0.0755148492)
    if k >= 66.0:
        b = 255.0
    elif k <= 19.0:
        b = 0.0
    else:
        b = 138.5177312231 * np.log(k - 10.0) - 305.0447927307
    rgb = np.array([r, g, b], dtype=np.float32)
    rgb = np.clip(rgb, 1.0, 255.0)
    # 转成乘数: 以最大通道为 1, 再取倒数 (色温高 -> 需要提升 R/B? 保持直觉一致性)
    mult = 255.0 / rgb
    mult = mult / float(np.exp(np.mean(np.log(mult))))
    return mult.astype(np.float32)


def gray_world_wb(img, low=1.0, high=99.0):
    """灰度世界白平衡: 在中间调区间估计通道均值比。"""
    lum = img.mean(axis=2)
    lo_v, hi_v = np.percentile(lum, low), np.percentile(lum, high)
    mask = (lum >= lo_v) & (lum <= hi_v) & (lum > 1e-5)
    if mask.sum() < 32:
        mask = lum > 1e-5
    means = np.array([img[..., c][mask].mean() for c in range(3)], dtype=np.float32)
    means = np.maximum(means, 1e-6)
    gains = means.mean() / means
    gains = gains / float(np.exp(np.mean(np.log(gains))))
    return gains.astype(np.float32)


# --------------------------------------------------------------------------
# 色温 / 白点 / Bradford 色适应 (用于手动色温白平衡)
# --------------------------------------------------------------------------

# 标准白点 xy 坐标
D65_XY = (0.31271, 0.32902)
D50_XY = (0.34567, 0.35850)


def xy_to_XYZ(xy, Y=1.0):
    """CIE xy -> XYZ (给定 Y)。"""
    x, y = xy
    if y <= 0:
        return np.array([0.0, Y, 0.0], dtype=np.float32)
    X = Y * x / y
    Z = Y * (1.0 - x - y) / y
    return np.array([X, Y, Z], dtype=np.float32)


def XYZ_to_xy(XYZ):
    """XYZ -> CIE xy。"""
    xyz = np.asarray(XYZ, dtype=np.float64)
    s = xyz.sum()
    if s <= 0:
        return (0.31271, 0.32902)
    return (float(xyz[0] / s), float(xyz[1] / s))


def kelvin_to_xy(temp_k):
    """色温(K) -> CIE xy 色度 (Krystek 1985 普朗克轨迹近似, 精度 <3e-4)。"""
    T = float(np.clip(temp_k, 1000.0, 40000.0))
    T2 = T * T
    T3 = T2 * T
    if T <= 4000.0:
        x = -0.2661239e9 / T3 - 0.2343589e6 / T2 + 0.8776956e3 / T + 0.179910
    else:
        x = -3.0258469e9 / T3 + 2.1070379e6 / T2 + 0.2226347e3 / T + 0.240390
    x = float(x)
    if T <= 2222.0:
        y = -1.1063814 * x ** 3 - 1.34811020 * x ** 2 + 2.18555832 * x - 0.20219683
    elif T <= 4000.0:
        y = -0.9549476 * x ** 3 - 1.37418593 * x ** 2 + 2.09137015 * x - 0.16748867
    else:
        y = 3.0817580 * x ** 3 - 5.87338670 * x ** 2 + 3.75112997 * x - 0.37001483
    return (x, float(np.clip(y, 0.0, 1.0)))


def bradford_CAT(src_xy, dst_xy):
    """Bradford 色适应矩阵: 把 src 白点下的 XYZ 映射到 dst 白点下。"""
    # Bradford 锥响应矩阵
    M_BFD = np.array([
        [0.8951, 0.2664, -0.1614],
        [-0.7502, 1.7135, 0.0367],
        [0.0389, -0.0685, 1.0296],
    ], dtype=np.float64)
    INV_BFD = np.linalg.inv(M_BFD)

    src = xy_to_XYZ(src_xy)
    dst = xy_to_XYZ(dst_xy)
    rho_src = M_BFD @ src
    rho_dst = M_BFD @ dst
    scale = rho_dst / rho_src
    M = INV_BFD @ np.diag(scale) @ M_BFD
    return M.astype(np.float32)


def adapt_xyz(xyz, src_xy, dst_xy):
    """对 XYZ 图像做 Bradford 色适应。xyz: ...x3 float32。"""
    M = bradford_CAT(src_xy, dst_xy)
    return (xyz.astype(np.float32) @ M.T).astype(np.float32)
