"""
RawForge 图形界面 (PySide6)。

布局:
    顶部工具栏 : 添加文件 / 添加目录 / 清空列表 / 输出目录
    中部三栏   : 文件列表 | 预览 | 参数面板(分页)
    底部       : 进度条 + 状态 + 开始处理

处理在后台线程执行, UI 不阻塞。支持拖放文件/目录到窗口。
"""

from __future__ import annotations

import os
import sys

from PySide6.QtCore import Qt, QThread, Signal, QSize, QUrl
from PySide6.QtGui import QPixmap, QDragEnterEvent, QDropEvent, QIcon
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QListWidget, QListWidgetItem, QLabel, QProgressBar,
    QTabWidget, QComboBox, QCheckBox, QSlider, QSpinBox, QLineEdit,
    QFileDialog, QSplitter, QGroupBox, QFormLayout, QToolBar, QStatusBar,
    QMessageBox, QScrollArea, QFrame,
)

from .core.pipeline import PipelineParams, RenderParams, OutputParams, DenoiseParams, LensCorrectionParams
from .core.pipeline import process_file, ProcessResult
from .core import io_utils

__all__ = ["MainWindow", "run"]


# --------------------------------------------------------------------------
# 后台处理线程
# --------------------------------------------------------------------------

class Worker(QThread):
    """后台批量处理线程。"""

    progress = Signal(int, int, str, float)   # (当前序号, 总数, 阶段名, 百分比)
    file_done = Signal(object)                 # ProcessResult
    all_done = Signal(list)                    # [ProcessResult, ...]
    failed = Signal(str)                       # 异常消息

    def __init__(self, files, params: PipelineParams, parent=None):
        super().__init__(parent)
        self.files = list(files)
        self.params = params

    def run(self):
        try:
            results = []
            total = len(self.files)
            for i, p in enumerate(self.files):
                def _cb(stage, pct, i=i):
                    self.progress.emit(i, total, stage, pct)
                r = process_file(p, self.params, _cb)
                results.append(r)
                self.file_done.emit(r)
            self.all_done.emit(results)
        except Exception as e:  # pragma: no cover
            self.failed.emit(str(e))


# --------------------------------------------------------------------------
# 参数控件辅助
# --------------------------------------------------------------------------

class FloatSlider(QWidget):
    """带标签的浮点滑块: 显示名 + 滑块 + 当前值标签。"""

    def __init__(self, label: str, lo: float, hi: float, value: float,
                 step: float = 0.05, fmt: str = "{:.2f}", parent=None):
        super().__init__(parent)
        self.fmt = fmt
        self._step = float(step) if step > 0 else 0.01
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self._name = QLabel(label)
        self._name.setMinimumWidth(72)
        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(int(round(lo / self._step)), int(round(hi / self._step)))
        self._slider.setValue(int(round(value / self._step)))
        self._val = QLabel(fmt.format(value))
        self._val.setMinimumWidth(40)
        self._val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        lay.addWidget(self._name)
        lay.addWidget(self._slider, 1)
        lay.addWidget(self._val)
        self._slider.valueChanged.connect(self._on_change)

    def _on_change(self, v):
        self._val.setText(self.fmt.format(self.value()))

    def value(self) -> float:
        return self._slider.value() * self._step

    def set_value(self, v: float):
        self._slider.setValue(int(round(v / self._step)))
        self._val.setText(self.fmt.format(v))


# --------------------------------------------------------------------------
# 主窗口
# --------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RawForge — 开源 RAW 处理")
        self.resize(1280, 800)
        self.setAcceptDrops(True)

        self._worker: Worker | None = None
        self._files: list[str] = []
        self._last_output: str = ""

        self._build_ui()

    # ------------------------------------------------------------------
    # UI 构建
    # ------------------------------------------------------------------

    def _build_ui(self):
        # 工具栏
        tb = QToolBar("操作")
        tb.setIconSize(QSize(18, 18))
        tb.setMovable(False)
        self.addToolBar(tb)
        self._act_add = tb.addAction("添加文件", lambda: self._add_files(False))
        self._act_add_dir = tb.addAction("添加目录", lambda: self._add_files(True))
        tb.addSeparator()
        self._act_clear = tb.addAction("清空", self._clear_files)
        tb.addSeparator()
        tb.addWidget(QLabel("  输出目录: "))
        self._out_dir = QLineEdit("")
        self._out_dir.setPlaceholderText("留空 = 与源文件同目录")
        self._out_dir.setMinimumWidth(260)
        tb.addWidget(self._out_dir)
        self._act_out = tb.addAction("浏览…", self._pick_out_dir)

        # 中央三栏
        split = QSplitter(Qt.Horizontal)

        # 左: 文件列表
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(4, 4, 4, 4)
        lv.addWidget(QLabel("文件队列"))
        self._list = QListWidget()
        self._list.setSelectionMode(QListWidget.ExtendedSelection)
        self._list.itemSelectionChanged.connect(self._on_select)
        lv.addWidget(self._list, 1)
        self._file_hint = QLabel("拖放 RAW 文件或文件夹到此处")
        self._file_hint.setStyleSheet("color: #999;")
        lv.addWidget(self._file_hint)
        split.addWidget(left)

        # 中: 预览
        mid = QWidget()
        mv = QVBoxLayout(mid)
        mv.setContentsMargins(4, 4, 4, 4)
        self._preview_title = QLabel("预览")
        self._preview_title.setAlignment(Qt.AlignCenter)
        mv.addWidget(self._preview_title)
        self._preview = QLabel("处理完成后在此显示结果")
        self._preview.setAlignment(Qt.AlignCenter)
        self._preview.setMinimumSize(480, 480)
        self._preview.setStyleSheet("background: #f5f5f5; color: #999; border: 1px solid #ddd;")
        self._preview.setScaledContents(False)
        mv.addWidget(self._preview, 1)
        split.addWidget(mid)

        # 右: 参数面板
        split.addWidget(self._build_params_panel())

        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 5)
        split.setStretchFactor(2, 3)
        split.setSizes([260, 640, 380])
        self.setCentralWidget(split)

        # 底部: 进度 + 开始
        bottom = QWidget()
        bv = QHBoxLayout(bottom)
        bv.setContentsMargins(8, 6, 8, 6)
        self._btn_start = QPushButton("开始处理")
        self._btn_start.setMinimumHeight(34)
        self._btn_start.clicked.connect(self._start)
        bv.addWidget(self._btn_start)
        self._progress = QProgressBar()
        self._progress.setTextVisible(True)
        bv.addWidget(self._progress, 1)
        self._stage_label = QLabel("就绪")
        self._stage_label.setMinimumWidth(120)
        bv.addWidget(self._stage_label)

        root = QWidget()
        rv = QVBoxLayout(root)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(0)
        rv.addWidget(split, 1)
        rv.addWidget(bottom)
        self.setCentralWidget(root)

        self.statusBar().showMessage("就绪 — 添加 RAW 文件后点击「开始处理」")

    def _build_params_panel(self) -> QWidget:
        tabs = QTabWidget()

        # ---- 解码 ----
        dec = QWidget()
        f = QFormLayout(dec)
        f.setContentsMargins(10, 10, 10, 10)
        self.c_demosaic = QComboBox()
        self.c_demosaic.addItems(["ahd", "amaze", "lmmse", "dcb", "ppg", "vng", "linear", "dht", "aahd"])
        self.c_demosaic.setCurrentText("ahd")
        f.addRow("去马赛克", self.c_demosaic)
        self.c_highlight = QComboBox()
        self.c_highlight.addItems(["blend", "clip", "reconstruct"])
        f.addRow("高光重建", self.c_highlight)
        self.c_half = QCheckBox("半尺寸快速解码(预览)")
        f.addRow("", self.c_half)
        tabs.addTab(dec, "解码")

        # ---- 镜头 ----
        lens = QWidget()
        f = QFormLayout(lens)
        f.setContentsMargins(10, 10, 10, 10)
        self.c_lens_dist = QCheckBox("畸变校正")
        self.c_lens_dist.setChecked(True)
        self.c_lens_tca = QCheckBox("横向色差校正")
        self.c_lens_tca.setChecked(True)
        self.c_lens_vig = QCheckBox("暗角校正")
        self.c_lens_vig.setChecked(True)
        self.c_lens_auto_tca = QCheckBox("自动估计色差(无数据库时)")
        self.c_lens_auto_vig = QCheckBox("自动估计暗角(无数据库时)")
        f.addRow(self.c_lens_dist)
        f.addRow(self.c_lens_tca)
        f.addRow(self.c_lens_vig)
        f.addRow(self.c_lens_auto_tca)
        f.addRow(self.c_lens_auto_vig)
        note = QLabel("提示: 优先使用 lensfun 光学数据库(948 机身/196 镜头)。\n未收录的机身自动回退参数化模型。")
        note.setWordWrap(True)
        note.setStyleSheet("color: #888;")
        f.addRow(note)
        tabs.addTab(lens, "镜头")

        # ---- 降噪 ----
        dn = QWidget()
        f = QFormLayout(dn)
        f.setContentsMargins(10, 10, 10, 10)
        self.c_dn_on = QCheckBox("启用降噪")
        self.c_dn_on.setChecked(True)
        f.addRow("", self.c_dn_on)
        self.c_dn_method = QComboBox()
        self.c_dn_method.addItems(["hybrid", "nlm", "wavelet", "bilateral", "median"])
        f.addRow("算法", self.c_dn_method)
        self.s_dn_strength = FloatSlider("强度", 0.0, 2.0, 1.0, 0.05)
        f.addRow("强度", self.s_dn_strength)
        self.s_dn_luma = FloatSlider("亮度", 0.0, 2.0, 1.0, 0.05)
        f.addRow("亮度降噪", self.s_dn_luma)
        self.s_dn_chroma = FloatSlider("色度", 0.0, 3.0, 0.6, 0.05)
        f.addRow("色度降噪", self.s_dn_chroma)
        self.s_dn_detail = FloatSlider("细节保护", 0.0, 1.0, 0.35, 0.01)
        f.addRow("细节保护", self.s_dn_detail)
        self.s_dn_sharpen = FloatSlider("锐化", 0.0, 1.0, 0.0, 0.01)
        f.addRow("降噪后锐化", self.s_dn_sharpen)
        tabs.addTab(dn, "降噪")

        # ---- 渲染 ----
        rend = QWidget()
        f = QFormLayout(rend)
        f.setContentsMargins(10, 10, 10, 10)
        self.s_r_exp = FloatSlider("曝光 EV", -3.0, 3.0, 0.0, 0.1)
        f.addRow("曝光补偿", self.s_r_exp)
        self.c_r_wb = QComboBox()
        self.c_r_wb.addItems(["camera", "auto", "temp", "none"])
        f.addRow("白平衡", self.c_r_wb)
        self.s_r_temp = FloatSlider("色温 K", 2000.0, 10000.0, 5500.0, 50.0, "{:.0f}")
        f.addRow("色温", self.s_r_temp)
        self.s_r_sat = FloatSlider("饱和度", 0.0, 2.0, 1.0, 0.05)
        f.addRow("饱和度", self.s_r_sat)
        self.s_r_con = FloatSlider("对比度", -1.0, 1.0, 0.0, 0.02)
        f.addRow("对比度", self.s_r_con)
        self.s_r_hl = FloatSlider("高光滚降", 0.0, 1.0, 0.0, 0.01)
        f.addRow("高光滚降", self.s_r_hl)
        self.s_r_sh = FloatSlider("暗部提亮", 0.0, 1.0, 0.0, 0.01)
        f.addRow("暗部提亮", self.s_r_sh)
        tabs.addTab(rend, "渲染")

        # ---- 输出 ----
        out = QWidget()
        f = QFormLayout(out)
        f.setContentsMargins(10, 10, 10, 10)
        self.c_out_fmt = QComboBox()
        self.c_out_fmt.addItems(["jpg", "png8", "png16", "tiff16", "dng16"])
        self.c_out_fmt.setCurrentText("jpg")
        f.addRow("格式", self.c_out_fmt)
        self.s_out_q = FloatSlider("质量", 1.0, 100.0, 95.0, 1.0, "{:.0f}")
        f.addRow("JPEG 质量", self.s_out_q)
        self.c_out_suffix = QLineEdit("_rawforge")
        f.addRow("文件名后缀", self.c_out_suffix)
        self.c_out_overwrite = QCheckBox("覆盖同名文件")
        f.addRow("", self.c_out_overwrite)
        tabs.addTab(out, "输出")

        return tabs

    # ------------------------------------------------------------------
    # 文件管理
    # ------------------------------------------------------------------

    def _add_files(self, directory: bool):
        if directory:
            d = QFileDialog.getExistingDirectory(self, "选择目录")
            if not d:
                return
            raws, imgs = io_utils.list_images([d])
            paths = raws + imgs
        else:
            paths, _ = QFileDialog.getOpenFileNames(
                self, "选择 RAW/图像文件", "",
                "图像 (*.cr2 *.cr3 *.nef *.arw *.raf *.dng *.orf *.rw2 *.pef *.srw *.x3f *.raw *.tif *.tiff *.jpg *.jpeg *.png)")
        for p in paths:
            if p not in self._files:
                self._files.append(p)
                self._list.addItem(os.path.basename(p))
        self._update_hint()

    def _clear_files(self):
        self._files.clear()
        self._list.clear()
        self._update_hint()

    def _update_hint(self):
        n = len(self._files)
        self._file_hint.setText(f"共 {n} 个文件 — 拖放 RAW 文件或文件夹到此处")
        self.statusBar().showMessage(f"已载入 {n} 个文件")

    def _on_select(self):
        pass  # 预留: 可在此加载源图预览

    def _pick_out_dir(self):
        d = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if d:
            self._out_dir.setText(d)

    # ------------------------------------------------------------------
    # 拖放
    # ------------------------------------------------------------------

    def dragEnterEvent(self, e: QDragEnterEvent):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e: QDropEvent):
        for url in e.mimeData().urls():
            p = url.toLocalFile()
            if not p:
                continue
            if os.path.isdir(p):
                raws, imgs = io_utils.list_images([p])
                paths = raws + imgs
            else:
                paths = [p]
            for q in paths:
                if q not in self._files:
                    self._files.append(q)
                    self._list.addItem(os.path.basename(q))
        self._update_hint()
        e.acceptProposedAction()

    # ------------------------------------------------------------------
    # 参数收集
    # ------------------------------------------------------------------

    def _collect_params(self) -> PipelineParams:
        P = PipelineParams()
        P.demosaic = self.c_demosaic.currentText()
        P.highlight_mode = self.c_highlight.currentText()
        P.half_size = self.c_half.isChecked()

        L = P.lens
        L.enable_distortion = self.c_lens_dist.isChecked()
        L.enable_tca = self.c_lens_tca.isChecked()
        L.enable_vignetting = self.c_lens_vig.isChecked()
        L.auto_tca = self.c_lens_auto_tca.isChecked()
        L.auto_vignetting = self.c_lens_auto_vig.isChecked()

        D = P.denoise
        D.enable = self.c_dn_on.isChecked()
        D.method = self.c_dn_method.currentText()
        D.strength = self.s_dn_strength.value()
        D.luminance = self.s_dn_luma.value()
        D.chroma = self.s_dn_chroma.value()
        D.detail_protect = self.s_dn_detail.value()
        D.sharpen_after = self.s_dn_sharpen.value()

        R = P.render
        R.exposure_ev = self.s_r_exp.value()
        R.white_balance = self.c_r_wb.currentText()
        R.wb_temp = self.s_r_temp.value()
        R.saturation = self.s_r_sat.value()
        R.contrast = self.s_r_con.value()
        R.highlight_rolloff = self.s_r_hl.value()
        R.shadow_lift = self.s_r_sh.value()

        O = P.output
        O.format = self.c_out_fmt.currentText()
        O.quality = int(self.s_out_q.value())
        O.suffix = self.c_out_suffix.text().strip() or "_rawforge"
        O.output_dir = self._out_dir.text().strip()
        O.overwrite = self.c_out_overwrite.isChecked()

        return P

    # ------------------------------------------------------------------
    # 处理
    # ------------------------------------------------------------------

    def _start(self):
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.information(self, "处理中", "已有任务在运行, 请等待完成。")
            return
        if not self._files:
            QMessageBox.warning(self, "无文件", "请先添加 RAW 或图像文件。")
            return

        params = self._collect_params()
        self._btn_start.setEnabled(False)
        self._act_add.setEnabled(False)
        self._act_add_dir.setEnabled(False)
        self._act_clear.setEnabled(False)
        self._progress.setRange(0, 1000)
        self._progress.setValue(0)
        self._stage_label.setText("启动…")

        self._worker = Worker(self._files, params, self)
        self._worker.progress.connect(self._on_progress)
        self._worker.file_done.connect(self._on_file_done)
        self._worker.all_done.connect(self._on_all_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_progress(self, idx, total, stage, pct):
        overall = (idx + pct) / max(total, 1)
        self._progress.setValue(int(overall * 1000))
        stage_name = dict([("decode", "解码"), ("profile", "噪声标定"), ("lens", "镜头校正"),
                           ("color", "色彩转换"), ("denoise", "降噪"), ("render", "渲染"),
                           ("write", "输出")]).get(stage, stage)
        self._stage_label.setText(f"{stage_name} {pct*100:.0f}%")

    def _on_file_done(self, r: ProcessResult):
        if r.ok and r.output:
            self._last_output = r.output
            self.statusBar().showMessage(f"完成: {os.path.basename(r.output)} ({r.elapsed:.1f}s)")

    def _on_all_done(self, results):
        self._progress.setValue(1000)
        self._stage_label.setText("完成")
        ok = sum(1 for r in results if r.ok)
        fail = len(results) - ok
        self.statusBar().showMessage(f"处理完成: 成功 {ok} / 失败 {fail}")
        self._reset_controls()
        # 预览最后成功的输出
        for r in reversed(results):
            if r.ok and r.output:
                self._show_preview(r.output)
                break
        if fail:
            msgs = [r.message for r in results if not r.ok]
            QMessageBox.warning(self, "部分失败", "\n".join(msgs[:5]))

    def _on_failed(self, msg):
        self._stage_label.setText("出错")
        self._reset_controls()
        QMessageBox.critical(self, "错误", msg)

    def _reset_controls(self):
        self._btn_start.setEnabled(True)
        self._act_add.setEnabled(True)
        self._act_add_dir.setEnabled(True)
        self._act_clear.setEnabled(True)

    def _show_preview(self, path: str):
        if not path or not os.path.exists(path):
            return
        pm = QPixmap(path)
        if pm.isNull():
            return
        label = self._preview
        scaled = pm.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaled)
        label.setStyleSheet("background: #222; border: 1px solid #ddd;")
        self._preview_title.setText(f"预览 — {os.path.basename(path)}")

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if getattr(self, "_last_output", "") and os.path.exists(self._last_output):
            self._show_preview(self._last_output)


# --------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------

def run():
    app = QApplication(sys.argv)
    app.setApplicationName("RawForge")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run()
