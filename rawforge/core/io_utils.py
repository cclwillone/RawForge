"""
IO 工具: 全面解决 Windows 中文/空格/长路径问题。

关键点:
    cv2.imread / cv2.imwrite 在 Windows 上无法处理非 ASCII 路径 (底层 C API 限制)。
    统一走 np.fromfile -> cv2.imdecode -> ... -> cv2.imencode -> ndarray.tofile,
    这条链路完全在 Python 层, 对中文路径、UNC 路径、超长路径均安全。
"""

from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path

import cv2
import numpy as np

__all__ = [
    "imread", "imwrite", "list_images", "unique_path", "safe_name",
    "is_raw", "RAW_EXTENSIONS", "IMAGE_EXTENSIONS", "ensure_dir",
    "human_size",
]


RAW_EXTENSIONS = {
    # 各厂商 RAW
    ".cr2", ".cr3", ".crw",          # Canon
    ".nef", ".nrw",                  # Nikon
    ".arw", ".srf", ".sr2",          # Sony
    ".orf",                          # Olympus
    ".rw2", ".raw", ".rwl",          # Panasonic / Leica
    ".raf",                          # Fujifilm
    ".pef", ".ptx", ".dcr",          # Pentax
    ".dng",                          # Adobe / 通用
    ".x3f", ".3fr", ".fff", ".iiq",  # Sigma / Hasselblad / PhaseOne
    ".erf", ".mdc", ".mos", ".mrw",
    ".k25", ".kdc", ".nrw", ".rwz",
    ".srf", ".srw", ".mef", ".cap",
}

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".jpe", ".png", ".tif", ".tiff", ".bmp", ".webp", ".jp2",
}


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)
    return str(p)


def is_raw(path) -> bool:
    return os.path.splitext(str(path))[1].lower() in RAW_EXTENSIONS


def _normalize(path):
    p = str(path)
    # 处理 macOS/NFD 与 Windows 长路径前缀
    p = unicodedata.normalize("NFC", p)
    p = os.path.abspath(p)
    if os.name == "nt" and not p.startswith("\\\\?\\") and len(p) > 240:
        p = "\\\\?\\" + p
    return p


def imread(path, flags=cv2.IMREAD_UNCHANGED):
    """读取普通图像 (支持中文路径)。返回 BGR/BGRA ndarray, 失败返回 None。"""
    p = _normalize(path)
    if not os.path.isfile(p):
        return None
    try:
        buf = np.fromfile(p, dtype=np.uint8)
        if buf.size == 0:
            return None
        img = cv2.imdecode(buf, flags)
        return img
    except Exception:
        return None


def imwrite(path, img, params=None) -> bool:
    """写出图像 (支持中文路径)。params 如 [cv2.IMWRITE_JPEG_QUALITY, 95]。"""
    p = _normalize(path)
    ensure_dir(os.path.dirname(p))
    ext = os.path.splitext(p)[1].lower()
    if not ext:
        p += ".png"
        ext = ".png"
    try:
        ok, buf = cv2.imencode(ext, img, params or [])
        if not ok:
            return False
        buf.tofile(p)
        return True
    except Exception:
        return False


def list_images(paths):
    """展开目录/文件混合输入, 返回去重后的 (RAW列表, 普通图列表)。"""
    raws, imgs = [], []
    seen = set()

    def add(p, raw):
        key = os.path.normcase(os.path.abspath(p))
        if key in seen:
            return
        seen.add(key)
        (raws if raw else imgs).append(p)

    for item in paths:
        p = str(item)
        if os.path.isdir(p):
            for root, _dirs, files in os.walk(p):
                for f in sorted(files):
                    ext = os.path.splitext(f)[1].lower()
                    if ext in RAW_EXTENSIONS:
                        add(os.path.join(root, f), True)
                    elif ext in IMAGE_EXTENSIONS:
                        add(os.path.join(root, f), False)
        elif os.path.isfile(p):
            ext = os.path.splitext(p)[1].lower()
            if ext in RAW_EXTENSIONS or ext in IMAGE_EXTENSIONS:
                add(p, ext in RAW_EXTENSIONS)
    return raws, imgs


def safe_name(name, maxlen=80):
    """生成合法文件名 (Windows 非法字符替换)。"""
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(name))
    name = name.strip().rstrip(". ")
    return name[:maxlen] if len(name) > maxlen else name


def unique_path(path):
    """若文件已存在, 追加 _1, _2 ... 避免覆盖。"""
    p = Path(path)
    if not p.exists():
        return str(p)
    stem, suffix, parent = p.stem, p.suffix, p.parent
    for i in range(1, 10000):
        cand = parent / f"{stem}_{i}{suffix}"
        if not cand.exists():
            return str(cand)
    return str(path)


def human_size(nbytes):
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024 or unit == "GB":
            return f"{nbytes:.1f} {unit}" if unit != "B" else f"{int(nbytes)} B"
        nbytes /= 1024.0
    return f"{nbytes:.1f} GB"
