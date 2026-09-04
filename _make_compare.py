"""
生成对比图:
    compare/_W7A0065_compare_base.png    引擎基础版 (rawpy vs RawForge 无降噪无校正)
    compare/_W7A0065_compare_full.png    完整流水线 (rawpy vs RawForge 含降噪+校正)
    compare/_W7A0065_compare_crop.png    中心 1:1 局部放大 (完整版)

DxO PureRAW 对标: 商业软件本机未安装, 无法实跑; 本对比图聚焦 rawpy 基线
与 RawForge 自身的对比, DxO 定位为"对标目标"在 README/说明里阐述。
"""
import os
import sys
import time

import numpy as np
import rawpy
import cv2
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rawforge.core.pipeline import RawPipeline, PipelineParams

SRC = r"C:\Users\admin\Downloads\SETUP_INSTALLER_(PASS=3450)\SETUP_INSTALLER\INSTALLER\_W7A0065.CR3"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compare")
os.makedirs(OUT, exist_ok=True)


def load_font(size):
    for p in (r"C:\Windows\Fonts\msyhbd.ttc", r"C:\Windows\Fonts\msyh.ttc",
              r"C:\Windows\Fonts\simhei.ttf"):
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


def rawpy_native(path):
    """rawpy/LibRaw 原生直出: 相机白平衡 + sRGB + 自动亮度, 无镜头校正、无降噪。"""
    with rawpy.imread(path) as raw:
        rgb = raw.postprocess(
            use_camera_wb=True,
            output_color=rawpy.ColorSpace.sRGB,
            output_bps=8,
            no_auto_bright=False,
            demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
        )
    return rgb


def rawforge(path, base_only=False):
    """RawForge 流水线。base_only=True 时关闭降噪与镜头校正, 仅跑基础管线。"""
    P = PipelineParams()
    P.output.output_dir = OUT
    P.output.format = "jpg"
    P.output.quality = 95
    P.output.suffix = "_base" if base_only else "_rawforge"
    P.output.overwrite = True
    if base_only:
        P.denoise.enable = False
        P.lens.enable_distortion = False
        P.lens.enable_tca = False
        P.lens.enable_vignetting = False
    r = RawPipeline(P).process(path)
    if not r.ok:
        raise RuntimeError(r.message)
    return np.array(Image.open(r.output).convert("RGB")), r


def stack_h(imgs, labels, gap=16, pad=20, bg=(245, 245, 245),
            font=None, label_h=54):
    if font is None:
        font = load_font(30)
    h = min(i.shape[0] for i in imgs)
    scaled = []
    for i in imgs:
        if i.shape[0] != h:
            w = int(round(i.shape[1] * h / i.shape[0]))
            i = cv2.resize(i, (w, h), interpolation=cv2.INTER_AREA)
        scaled.append(i)
    widths = [i.shape[1] for i in scaled]
    total_w = pad * 2 + sum(widths) + gap * (len(imgs) - 1)
    canvas = np.full((pad * 2 + label_h + h, total_w, 3), bg[0], dtype=np.uint8)
    canvas = Image.fromarray(canvas)
    d = ImageDraw.Draw(canvas)
    x = pad
    for img, lab in zip(scaled, labels):
        y = pad + label_h
        canvas.paste(Image.fromarray(img), (x, y))
        d.text((x, pad + 8), lab, fill=(30, 30, 30), font=font)
        x += img.shape[1] + gap
    return np.array(canvas)


def main():
    print("== 1/4 rawpy 原生直出 ==")
    t0 = time.time()
    ref = rawpy_native(SRC)
    print("   耗时 %.1fs, 尺寸 %s" % (time.time() - t0, ref.shape))

    print("== 2/4 RawForge 基础版 (引擎管线验证) ==")
    t0 = time.time()
    base, base_res = rawforge(SRC, base_only=True)
    print("   耗时 %.1fs" % (time.time() - t0))

    print("== 3/4 RawForge 完整版 (降噪+镜头校正) ==")
    t0 = time.time()
    full, full_res = rawforge(SRC, base_only=False)
    print("   耗时 %.1fs" % (time.time() - t0))
    print("   镜头: %s" % (full_res.lens_info.message if full_res.lens_info else "n/a"))

    h = min(ref.shape[0], base.shape[0], full.shape[0])
    w = min(ref.shape[1], base.shape[1], full.shape[1])
    ref = ref[:h, :w]; base = base[:h, :w]; full = full[:h, :w]

    fw = 1200
    scale = fw / w
    ref_s = cv2.resize(ref, (fw, int(h * scale)), interpolation=cv2.INTER_AREA)
    base_s = cv2.resize(base, (fw, int(h * scale)), interpolation=cv2.INTER_AREA)
    full_s = cv2.resize(full, (fw, int(h * scale)), interpolation=cv2.INTER_AREA)

    canvas = stack_h([ref_s, base_s],
                     ["rawpy 原生直出 (基线)",
                      "RawForge 引擎基础 (解码+色彩+渲染)"])
    Image.fromarray(canvas).save(os.path.join(OUT, "_W7A0065_compare_base.png"))
    print("   已生成: _W7A0065_compare_base.png  (主图: 引擎基础)")

    canvas2 = stack_h([ref_s, full_s],
                      ["rawpy 原生直出 (基线)",
                       "RawForge 完整流水线 (lensfun + 降噪)"])
    Image.fromarray(canvas2).save(os.path.join(OUT, "_W7A0065_compare_full.png"))
    print("   已生成: _W7A0065_compare_full.png  (完整版)")

    cw, ch = 900, 620
    cx, cy = w // 2 - cw // 2, h // 2 - ch // 2
    ref_c = ref[cy:cy + ch, cx:cx + cw]
    full_c = full[cy:cy + ch, cx:cx + cw]
    crop = stack_h([ref_c, full_c],
                   ["rawpy 原生 — 中心 100%",
                    "RawForge 完整 — 中心 100%"],
                   font=load_font(22), label_h=40, gap=12)
    Image.fromarray(crop).save(os.path.join(OUT, "_W7A0065_compare_crop.png"))
    print("   已生成: _W7A0065_compare_crop.png  (1:1 局部)")

    print("== 完成 ==")


if __name__ == "__main__":
    main()