"""端到端测试: Worker 处理真实 CR3, 验证信号链路 + 输出。"""
import os, sys, time
os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, ".")

from PySide6.QtWidgets import QApplication
from rawforge.gui import Worker
from rawforge.core.pipeline import PipelineParams

app = QApplication(sys.argv)

cr3 = r"C:\Users\admin\Downloads\SETUP_INSTALLER_(PASS=3450)\SETUP_INSTALLER\INSTALLER\_W7A0065.CR3"
P = PipelineParams()
P.output.output_dir = "C:/Users/admin/Downloads/RawForge/out"
P.output.format = "jpg"
P.output.quality = 90
P.output.suffix = "_gui"

w = Worker([cr3], P)
events = []
w.progress.connect(lambda i, t, s, p: events.append(("progress", s, round(p, 2))))
w.file_done.connect(lambda r: events.append(("done", r.ok, r.output)))
w.all_done.connect(lambda rs: events.append(("all", len(rs))))

t0 = time.time()
w.run()  # 同步执行 (不 start 线程)
dt = time.time() - t0

print("耗时: %.1fs" % dt)
print("信号事件总数: %d" % len(events))
prog = [e for e in events if e[0] == "progress"]
done = [e for e in events if e[0] == "done"]
all_ = [e for e in events if e[0] == "all"]
print("progress 事件: %d 次 (阶段: %s)" % (len(prog), sorted(set(e[1] for e in prog))))
print("done 事件: ", done)
print("all 事件: ", all_)

assert all_, "缺少 all_done 信号"
assert done and done[0][1] is True, "处理失败"
out = done[0][2]
assert os.path.exists(out), "输出文件不存在: %s" % out
print("输出文件存在: %s (%.2f MB)" % (out, os.path.getsize(out) / 1024 / 1024))
print("\n[PASS] 端到端测试通过")
