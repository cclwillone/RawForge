"""
RawForge 命令行接口。

用法示例:
    # 处理单张 RAW, 输出 16bit TIFF
    python -m rawforge.cli photo.cr3

    # 批量处理整个目录, 输出 JPEG
    python -m rawforge.cli ./photos --format jpg --quality 92

    # 只降噪, 输出 Linear DNG 交给 Lightroom 继续调色
    python -m rawforge.cli photo.cr3 --mode denoise_only --format dng16

    # 手动色温 5500K, 提亮高光, 加强降噪
    python -m rawforge.cli photo.cr3 --wb temp --temp 5500 --highlight-rolloff 0.4 --strength 1.5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from .core.pipeline import PipelineParams, RenderParams, OutputParams, DenoiseParams, LensCorrectionParams
from .core.pipeline import process_file, process_batch
from .core import io_utils


def build_parser():
    p = argparse.ArgumentParser(
        prog="rawforge",
        description="开源 RAW 处理流水线 (DxO PureRAW 的免费替代)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("inputs", nargs="+", help="RAW/图像文件或目录")

    # 输出
    g = p.add_argument_group("输出")
    g.add_argument("--output-dir", "-o", default="", help="输出目录 (默认与源文件同目录)")
    g.add_argument("--format", "-f", default="tiff16",
                   choices=["tiff16", "png16", "png8", "jpg", "dng16"],
                   help="输出格式")
    g.add_argument("--mode", default="full", choices=["full", "denoise_only"],
                   help="full=完整出片; denoise_only=只降噪输出线性DNG/TIFF")
    g.add_argument("--quality", type=int, default=95, help="JPEG 质量 1-100")
    g.add_argument("--suffix", default="_rawforge", help="输出文件名后缀")
    g.add_argument("--overwrite", action="store_true", help="覆盖同名文件")
    g.add_argument("--workers", type=int, default=1, help="并行处理线程数")

    # 解码
    g = p.add_argument_group("解码")
    g.add_argument("--demosaic", default="ahd",
                   choices=["linear", "vng", "ppg", "ahd", "dcb", "amaze", "lmmse", "dht", "aahd"],
                   help="去马赛克算法")
    g.add_argument("--highlight", default="blend", choices=["clip", "blend", "reconstruct"],
                   help="高光重建模式")
    g.add_argument("--half-size", action="store_true", help="半尺寸快速解码 (预览用)")

    # 镜头校正
    g = p.add_argument_group("镜头校正")
    g.add_argument("--no-lens", action="store_true", help="关闭全部镜头校正")
    g.add_argument("--no-distortion", action="store_true", help="关闭畸变校正")
    g.add_argument("--no-tca", action="store_true", help="关闭横向色差校正")
    g.add_argument("--no-vignetting", action="store_true", help="关闭暗角校正")
    g.add_argument("--auto-tca", action="store_true", help="自动估计色差")
    g.add_argument("--auto-vignetting", action="store_true", help="自动估计暗角")

    # 降噪
    g = p.add_argument_group("降噪")
    g.add_argument("--no-denoise", action="store_true", help="关闭降噪")
    g.add_argument("--strength", type=float, default=1.0, help="降噪强度倍率")
    g.add_argument("--method", default="hybrid",
                   choices=["hybrid", "nlm", "wavelet", "bilateral", "median"],
                   help="降噪算法")
    g.add_argument("--luminance", type=float, default=1.0, help="亮度降噪强度")
    g.add_argument("--chroma", type=float, default=0.6, help="色度降噪强度")
    g.add_argument("--sharpen", type=float, default=0.0, help="降噪后锐化 0~1")

    # 渲染
    g = p.add_argument_group("渲染")
    g.add_argument("--exposure", type=float, default=0.0, help="曝光补偿 EV")
    g.add_argument("--wb", default="camera", choices=["camera", "auto", "temp", "manual", "none"],
                   help="白平衡模式")
    g.add_argument("--temp", type=float, default=5500.0, help="手动色温 (K)")
    g.add_argument("--saturation", type=float, default=1.0, help="饱和度 0~2")
    g.add_argument("--contrast", type=float, default=0.0, help="对比度 -1~1")
    g.add_argument("--highlight-rolloff", type=float, default=0.0, help="高光滚降 0~1")
    g.add_argument("--shadow-lift", type=float, default=0.0, help="暗部提亮 0~1")
    g.add_argument("--output-space", default="srgb",
                   choices=["srgb", "adobe_rgb", "prophoto", "rec2020"], help="输出色彩空间")

    # 其它
    g = p.add_argument_group("其它")
    g.add_argument("--dump-params", action="store_true", help="打印当前参数后退出")
    g.add_argument("--json", action="store_true", help="结果以 JSON 输出")
    return p


def _build_params(args) -> PipelineParams:
    P = PipelineParams()
    P.demosaic = args.demosaic
    P.highlight_mode = args.highlight
    P.half_size = args.half_size

    L = P.lens
    if args.no_lens:
        L.enable_distortion = L.enable_tca = L.enable_vignetting = False
    if args.no_distortion:
        L.enable_distortion = False
    if args.no_tca:
        L.enable_tca = False
    if args.no_vignetting:
        L.enable_vignetting = False
    L.auto_tca = args.auto_tca
    L.auto_vignetting = args.auto_vignetting

    D = P.denoise
    D.enable = not args.no_denoise
    D.strength = args.strength
    D.method = args.method
    D.luminance = args.luminance
    D.chroma = args.chroma
    D.sharpen_after = args.sharpen

    R = P.render
    R.exposure_ev = args.exposure
    R.white_balance = args.wb
    R.wb_temp = args.temp
    R.saturation = args.saturation
    R.contrast = args.contrast
    R.highlight_rolloff = args.highlight_rolloff
    R.shadow_lift = args.shadow_lift
    R.output_space = args.output_space

    O = P.output
    O.mode = args.mode
    O.format = args.format
    O.quality = args.quality
    O.output_dir = args.output_dir
    O.suffix = args.suffix
    O.overwrite = args.overwrite

    return P


def main(argv=None):
    args = build_parser().parse_args(argv)
    params = _build_params(args)

    if args.dump_params:
        print(json.dumps(params.to_dict(), ensure_ascii=False, indent=2))
        return 0

    raws, imgs = io_utils.list_images(args.inputs)
    all_files = raws + imgs
    if not all_files:
        print("未找到任何 RAW/图像文件", file=sys.stderr)
        return 2

    print(f"待处理 {len(all_files)} 个文件 (RAW {len(raws)}, 普通图 {len(imgs)})")
    if not params.denoise.enable:
        print("提示: 降噪已关闭")

    def progress(stage, pct, idx=None, total=None):
        if idx is not None:
            name = io_utils.list_images.__name__
            bar = "#" * int(pct * 30) + "-" * (30 - int(pct * 30))
            print(f"\r[{idx}/{total}] {bar} {pct*100:5.1f}%", end="", flush=True)
        else:
            pass

    results = process_batch(all_files, params, workers=max(1, args.workers), progress=progress)
    if args.workers > 1:
        print()

    # 汇总
    ok = [r for r in results if r.ok]
    fail = [r for r in results if not r.ok]
    print("\n" + "=" * 60)
    print(f"完成: 成功 {len(ok)} / 失败 {len(fail)}")
    if args.json:
        out = [{
            "source": r.source, "output": r.output, "ok": r.ok,
            "message": r.message, "elapsed": round(r.elapsed, 3),
            "profile": r.profile.to_dict() if r.profile else None,
            "lens": r.lens_info.message if r.lens_info else None,
            "size": [r.width, r.height],
        } for r in results]
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        for r in results:
            print(r)
        for r in fail:
            print("  失败原因:", r.message)
    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main())
