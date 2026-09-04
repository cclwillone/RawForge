"""
RawForge - 开源 RAW 处理流水线 (DxO PureRAW 的免费替代方案)

纯 Python + 开源库实现，无任何商业依赖，适配 Windows 10/11。

管线:
    RAW 解码(线性域) -> 镜头校正 -> 降噪(噪声剖面自适应) -> 色彩渲染 -> 输出
"""

__version__ = "1.0.0"
__author__ = "RawForge"

__all__ = ["core", "cli"]
