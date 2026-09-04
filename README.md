# RawForge

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Windows%2010%2F11-0078D6?logo=windows&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-success)
![GUI](https://img.shields.io/badge/GUI-PySide6%20(Qt6)-41CD52?logo=qt&logoColor=white)
![Code](https://img.shields.io/badge/Code-~3.5k%20lines-blueviolet)

> **完全开源、免付费的 RAW 照片处理流水线**
> 对标 DxO PureRAW 的「光学校正 + 智能降噪」工作流，**核心算法全部自研**
> （numpy / OpenCV 向量化实现），适配 Windows 10 / 11。

[English](./README_EN.md) | **简体中文**

---

## ✨ 特性亮点

| | 特性 | 说明 |
|---|---|---|
| 🎯 | **噪声剖面自标定** | 无需为任何相机 / ISO 预校准。逐张从传感器光学遮罩区估计读出噪声、用高通残差 + MAD 分箱拟合散粒噪声，得到完整模型 `σ² = a·I + b`。实测估计误差 **< 3.5%**、置信度 **0.99** |
| 🔬 | **线性域降噪引擎** | 全程在**线性光**域降噪（gamma 域噪声非平稳，无法建模）。组合：向量化 **NLM 非局部均值**（`boxFilter` 块距离降到 O(1)）+ **Haar 小波循环平移软阈值**（消除 Gibbs 振铃）+ **亮度/色度分离**（DeepPRIME 同款策略）+ 细节保护掩膜 |
| 🎨 | **专业色彩管线** | 相机色矩阵 → XYZ → 目标色彩空间，Bradford 色适应手动色温。已绕开 LibRaw 色矩阵的两个经典陷阱，中性色精确落在 D65（色度偏差 < 0.02） |
| 🔍 | **光学镜头校正** | 畸变 / 横向色差 / 暗角三通道校正，基于 lensfun 数据库（948 机身）；新机身数据以**内嵌 XML** 方式补齐（已含 Canon EOS R6 Mark III / R5 Mark II），镜头型号从 RAW 元数据自动识别 |
| 🔄 | **双工作模式** | `full` 直出成片；`denoise_only` 输出 **Linear DNG**（DNG 1.4 + ColorMatrix + AsShotNeutral），保留全部后期空间交给 Lightroom / Capture One / darktable —— 复刻 PureRAW 的核心工作流 |
| 🖥 | **图形界面 + 命令行** | PySide6 三栏 GUI，支持拖放文件/文件夹批量处理，后台线程不卡界面；CLI 支持目录批量、JSON 结果输出，便于脚本化与 CI |

---

## 🖼 效果对比

同一张 RAW，默认参数 —— 上为 rawpy / LibRaw 原生直出基线，下为 RawForge 完整流水线：

<p align="center">
  <img src="compare/_W7A0065_compare_full.png" width="900" alt="RawForge 完整流水线 vs rawpy 原生直出">
  <br><em>整幅对比 · RawForge 完整流水线（lensfun 校正 + 线性域降噪）vs rawpy 原生直出</em>
</p>

<p align="center">
  <img src="compare/_W7A0065_compare_crop.png" width="900" alt="中心 1:1 局部对比">
  <br><em>中心 1:1 局部放大 —— 关注细密纹理保留与暗部彩噪抑制</em>
</p>

> 说明：DxO PureRAW 为商业软件，本项目未使用它生成对照样张，仅将其**工作流定位**作为
> 设计参照。RawForge 的降噪为传统算法实现（NLM + 小波），**不是深度学习模型**，
> 详见 [路线图](#-路线图)。

---

## 🚀 快速开始

### 环境要求

- Windows 10 / 11（x64）
- Python **3.13**（3.10+ 理论上可用，本项目在 3.13 上验证）
- 磁盘空间：源码版约 500 MB（含依赖），打包版约 280 MB

### 一键安装（推荐）

```bat
双击 setup_env.bat      :: 自动创建虚拟环境并安装依赖（清华镜像加速）
双击 run.bat            :: 启动图形界面
```

### 手动安装

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python main.py
```

### 直接使用打包版

从 Releases 下载 `RawForge.zip`，解压后整个目录一起拷贝，双击 `RawForge.exe` 即可
（无需安装 Python）。

---

## 💻 使用方式

### 图形界面

拖入 RAW 文件或文件夹 → 右侧调节参数（**解码 / 镜头 / 降噪 / 渲染 / 输出** 五个分页）
→ 点击「开始处理」→ 底部进度条实时显示阶段 → 完成后中间预览出片。

### 命令行

```bash
# 单张出 JPEG
python main.py --cli photo.cr3 --format jpg --quality 95

# 批量处理整个目录，输出 16bit TIFF（workers 并行）
python main.py --cli ./photos --format tiff16 --workers 2

# 只降噪，输出 Linear DNG 交给 Lightroom 继续调色
python main.py --cli photo.cr3 --mode denoise_only --format dng16

# 手动色温 5500K + 高光滚降 + 加强降噪
python main.py --cli photo.cr3 --wb temp --temp 5500 --highlight-rolloff 0.4 --strength 1.5

# 以 JSON 输出结果，便于脚本解析
python main.py --cli ./photos --format jpg --json
```

完整参数见 `python main.py --cli --help`。

---

## ⚙️ 工作原理

### 八阶段流水线

```
解码 → 噪声标定 → 镜头校正 → 白平衡 → 色彩转换 → 降噪 → 渲染 → 输出
```

| 阶段 | 说明 |
|---|---|
| **1 解码** | LibRaw 去马赛克（AHD / AMAZE / LMMSE / DCB 等可选），输出线性光相机原生 RGB |
| **2 噪声标定** | 从光学遮罩区 + 亮度分箱统计拟合泊松-高斯噪声剖面 `σ² = a·I + b` |
| **3 镜头校正** | lensfun 畸变 / 横向色差 / 暗角（必须在去马赛克后的线性域做） |
| **4 白平衡** | 相机空间乘通道增益；降噪**前**处理，否则蓝通道噪声比例失真 |
| **5 色彩转换** | 相机 RGB → XYZ → sRGB / Adobe RGB / ProPhoto / Rec.2020 |
| **6 降噪** | 亮度/色度分离，噪声剖面自适应强度 |
| **7 渲染** | 曝光 / 高光滚降 / 暗部提亮 / 对比度 S 曲线 / 饱和度 / gamma 编码 |
| **8 输出** | 16bit TIFF / PNG / JPEG(4:4:4) / Linear DNG |

### 为什么必须在线性域降噪

线性域噪声近似高斯且与信号强度可建模（`var = a·I + b`）；gamma 编码会压缩亮部、
拉伸暗部，使噪声变得**非平稳**，模型失效、阈值失去意义 —— 这是出片干净与否的根本分界。

### 向量化 NLM 的三个关键实现

1. **O(1) 块距离**：用 `cv2.boxFilter` 一次性算出全图层面的块级平方距离，完全向量化；
2. **亮度共享权重**：相似度由亮度通道统一计算后共享给 RGB，杜绝通道间颜色渗透；
3. **2σ² 无偏修正**：含噪块距离的期望自带 `2σ²·area` 偏置，扣除后避免高噪声下权重退化。

---

## 📊 实测数据

| 项目 | 结果 |
|---|---|
| 输入 | Canon EOS R6 Mark III · CR3 · 39.6 MB |
| 分辨率 | 6959 × 4639（3200 万像素） |
| 噪声标定 | 置信度 **0.99**，估计误差 **< 3.5%** |
| 镜头识别 | RF70-200mm F2.8 L IS USM（从 RAW 元数据自动提取） |
| EXIF | ISO 640 · 70mm · f/4.5 · 1/125s（CR3 BMFF 容器 → 缩略图回退提取） |
| 单张耗时 | 约 **65 s**（AHD 去马赛克 + hybrid 降噪，默认画质） |
| 输出 | 3.85 MB JPEG（quality 95） |
| 打包一致性 | exe 与源码输出 **MD5 逐字节一致** |

---

## 📁 项目结构

```
RawForge/
├── rawforge/
│   ├── core/
│   │   ├── decode.py      # RAW 解码 + EXIF（CR3 BMFF 缩略图回退）      ≈ 400 行
│   │   ├── noise.py       # 噪声剖面自标定                              ≈ 315 行
│   │   ├── denoise.py     # NLM + 小波 + 亮色分离降噪引擎                ≈ 480 行
│   │   ├── lens.py        # lensfun 校正 + 新机身后备数据 + 镜头匹配     ≈ 560 行
│   │   ├── color.py       # 色彩矩阵 / Bradford 色适应 / 传递函数        ≈ 355 行
│   │   ├── pipeline.py    # 八阶段编排 + 批量线程池 + Linear DNG 写入    ≈ 500 行
│   │   └── io_utils.py    # 中文路径安全 IO
│   ├── gui.py             # PySide6 图形界面                            ≈ 520 行
│   └── cli.py             # 命令行接口                                   ≈ 200 行
├── main.py                # 入口（GUI / --cli）
├── compare/               # 效果对比样张
├── samples/               # 示例 RAW（Nikon D7200 NEF）
├── requirements.txt       # 依赖清单
├── setup_env.bat          # 一键安装环境
├── run.bat                # 一键启动
├── RawForge.spec          # PyInstaller 打包配置（可复现构建）
├── DEVELOPMENT_SUMMARY.html  # 开发过程总结
├── LICENSE                # MIT
├── NOTICE                 # 第三方组件与合规声明
└── README.md / README_EN.md
```

---

## 🛣 路线图

- [ ] **AI 降噪**：接入开源 CNN 降噪网络，对标真·DeepPRIME（现有噪声剖面可直接作为训练目标）
- [ ] **性能优化**：65 s/张的瓶颈在 NLM 搜索窗，计划降分辨率分块或 OpenCL / CUDA 异构加速
- [ ] **实时预览**：`half_size` 快速解码接口已预留，接滑杆实时预览
- [ ] **相机覆盖**：镜头数据随 lensfun 上游同步，并支持用户自定义镜头 XML 导入
- [ ] **跨平台**：Linux / macOS 构建验证（当前代码无 Windows 专有依赖，仅 GUI 与打包未测）

欢迎提交 Issue 与 PR。

---

## 📄 许可与声明

### 项目许可

RawForge 自身源代码采用 **[MIT License](./LICENSE)** —— 可自由使用、修改、二次开发与商用。

### 第三方组件

项目依赖若干开源库，其中 **LibRaw (LGPL v2.1)**、**lensfun (LGPL v3)**、
**PySide6 (LGPL v3)** 在二进制分发中需遵守 LGPL 的可替换性与随附许可文本要求。
完整清单与合规说明见 **[NOTICE](./NOTICE)**。

> 若你分发自己的修改版二进制程序，请确保随附 `NOTICE` 与上述 LGPL 文本。

### 商标声明

**DxO**、**PureRAW**、**DeepPRIME** 为 DxO Labs 的商标；**Lightroom** 为 Adobe 的商标；
**Canon**、**EOS**、**RF** 为 Canon Inc. 的商标。本项目提及这些名称仅用于说明功能定位
与互操作性，**与上述公司无任何关联，未获授权或背书**，也未使用其任何专有代码、模型或数据。

### 免责声明

本软件按「**原样**」（AS IS）提供，不附带任何明示或暗示的担保。作者不对因使用本软件
导致的任何数据损失或处理效果负责。

- 处理过程中**不会修改或覆盖原始 RAW 文件**，但仍建议在处理前备份原始素材；
- 输出画质受拍摄条件、镜头数据覆盖情况与参数设置影响，实际效果请以样张测试为准；
- 本项目为个人开发的技术作品，非专业影像生产软件的替代品。

---

## 🙏 致谢

本项目的实现建立在以下开源项目之上，特此感谢：

[LibRaw](https://www.libraw.org/) · [lensfun](https://lensfun.github.io/) ·
[rawpy](https://github.com/letmaik/rawpy) · [lensfunpy](https://github.com/letmaik/lensfunpy) ·
[OpenCV](https://opencv.org/) · [NumPy](https://numpy.org/) ·
[PySide6](https://doc.qt.io/qtforpython/) · [Pillow](https://python-pillow.org/) ·
[tifffile](https://github.com/cgohlke/tifffile) · [ExifRead](https://github.com/ianare/exif-py)

---

<sub>若本项目对你有帮助，欢迎点个 ⭐ Star。</sub>
