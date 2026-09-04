"""GUI 离屏冒烟测试: 验证界面可构建、参数收集正确。"""
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import sys
sys.path.insert(0, ".")

from PySide6.QtWidgets import QApplication
from rawforge.gui import MainWindow, Worker
from rawforge.core.pipeline import PipelineParams

app = QApplication(sys.argv)
win = MainWindow()

# 1. 参数收集
P = win._collect_params()
print("=== 参数收集 ===")
print("demosaic:", P.demosaic, "| highlight:", P.highlight_mode, "| half:", P.half_size)
print("lens: dist=%s tca=%s vig=%s" % (P.lens.enable_distortion, P.lens.enable_tca, P.lens.enable_vignetting))
print("denoise: on=%s method=%s strength=%.2f luma=%.2f chroma=%.2f" % (
    P.denoise.enable, P.denoise.method, P.denoise.strength, P.denoise.luminance, P.denoise.chroma))
print("render: exp=%.2f wb=%s temp=%.0f sat=%.2f" % (
    P.render.exposure_ev, P.render.white_balance, P.render.wb_temp, P.render.saturation))
print("output: fmt=%s q=%d suffix=%s" % (P.output.format, P.output.quality, P.output.suffix))

# 2. 修改控件再收集, 验证双向绑定
win.c_dn_method.setCurrentText("wavelet")
win.s_dn_strength.set_value(1.5)
win.c_out_fmt.setCurrentText("tiff16")
P2 = win._collect_params()
assert P2.denoise.method == "wavelet", "method 绑定失败"
assert abs(P2.denoise.strength - 1.5) < 1e-6, "strength 绑定失败"
assert P2.output.format == "tiff16", "format 绑定失败"
print("\n双向绑定 OK: method=%s strength=%.2f fmt=%s" % (P2.denoise.method, P2.denoise.strength, P2.output.format))

# 3. Worker 信号存在性
w = Worker([], PipelineParams())
print("\nWorker 信号:", [s for s in dir(w) if s in ("progress", "file_done", "all_done", "failed")])

# 4. 模拟添加文件到队列
win._files.append(r"C:\fake\test.cr3")
win._list.addItem("test.cr3")
win._update_hint()
print("队列:", win._files, "| 状态:", win.statusBar().currentMessage())

print("\n[PASS] GUI 冒烟测试通过")
