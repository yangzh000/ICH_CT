# 取栓后脑出血鉴别：MATLAB + Python 方法复现

MATLAB 负责顺序前向特征选择（SFS）；Python 负责影像处理、影像组学、habitat 生成、SVM 概率预测和三维深度学习。实现依据为最先提供的 Radiology 稿件、Supplementary Methods v04，以及后续明确确认的参数。Python 与 MATLAB 源码均不含注释或文档字符串，使用说明集中在本文件。

本项目是依据文字方法重新编写的实现。目前未获得原始 CT、分割、患者标签、原始代码或模型权重，因此不能称为论文数值结果的复现。`config.study.json` 包含明确标出的实现参数，正式运行前应根据原研究记录核对。测试使用合成数据，测试结果不代表临床性能。

## 分工

| 模块 | 实现 |
|---|---|
| CT 与 ROI 配准检查、1 mm 重采样、灰度处理、训练集增强 | Python / SimpleITK、SciPy |
| 全病灶、局部 habitat 与 patch 特征 | Python / PyRadiomics |
| SFS | MATLAB / 自定义顺序前向搜索、`fitcsvm` 高斯核 SVM |
| 最终 SVM、训练内概率校准 | Python / scikit-learn、SciPy |
| OOF habitat、重叠 patch 概率平均、逐患者 K-means | Python |
| 三维 ResNet34、三维 Swin 分类器 | Python / PyTorch、MONAI |
| AUC、分类指标、DeLong、DCA | Python；MATLAB 独立计算同一预测表 |
| SHAP | Python / SHAP |

## 已确认的方法

- 开发队列为中心 A 的 299 例，外部队列为中心 B、C 共 125 例。标签为 HT = 1、CS = 0。
- 内部测试为患者级五折，重复五次。每个患者在每次重复中仅有一次测试预测，不选择表现最好的一次重复报告。
- SFS 在当前训练集内按交叉验证平均 AUC 选择特征子集。所有患者的增强版本随原患者进入同一训练分组，验证仅使用原始版本。
- SVM 使用高斯核。
- Patch 为重采样后的 20 × 20 × 20 体素，步长 1 体素；ROI 边界保留周围图像信息。模型只使用整个 ROI 的患者级标签，无 patch 标签或标签权重。
- 每轮由该轮初始全病灶/当前保留区域 SVM 直接预测 patch；没有另行用 patch 标签训练分类器。
- 重叠 patch 的 HT 概率在体素处取平均。每位患者每轮重新聚为 10 类，删除平均 HT 概率最低的一类，合并其余区域后重新训练局部 SVM。
- 训练患者的 habitat 使用 OOF 方式产生；递归交叉拟合从整个先前链条中排除被预测患者。
- ResNet34 和 Swin 均接收三维体积，采用单通道二分类网络。
- 内部测试完成后在全部 299 例开发患者上重新训练，外部患者仅用于冻结模型后的预测和评价。

原始 5 mm 层厚重建与人工分割属于输入数据的获取过程。本项目接收已重建的三维图像和已确认的二值 ROI，不从文稿推断人工标注，也不会先对已有图像再次模拟 5 mm 重建。

## 需要与原始记录核对的实现参数

下列取值用于提供可执行实现，不表示已经确认是原研究设置。未提供的信息均集中在配置文件中，可修改后在开发数据内重新运行；不能根据外部结果选择参数。

| 配置项 | 当前实现取值 |
|---|---|
| 图像插值、灰度标准化 | 线性插值；每个图像单独 z-score，无 HU 截断；ROI 用最近邻插值 |
| 灰度离散化与滤波 | 32 个灰度箱，仅 Original；未加入未经确认的 LoG / wavelet 滤波 |
| 影像组学类别 | shape、firstorder、GLCM、GLRLM、GLSZM、GLDM、NGTDM；类别内启用 PyRadiomics 默认非弃用特征 |
| 后续轮次与最终 habitat 特征 | texture，依据最先稿件中的描述；配置可切换为 all |
| 训练增强 | 每人增加 1 个版本；旋转 ±10°、缩放 0.9–1.1、裁剪边距 0–5 体素；保留全部变换后 ROI |
| SVM | C = 1；gamma = scale；不做额外超参数搜索 |
| SFS 搜索 | 贪心逐一加入特征，默认遍历所有候选数量，保存各数量平均 AUC；选取路径上平均 AUC 最高的子集，平分时选较短子集 |
| 各内部交叉验证 | SFS、概率校准、habitat OOF、深度选择、阈值选择均默认五折；SFS 内部重复默认 1 次，与外部五折测试重复 5 次分开 |
| 概率校准 | 基于训练集分组 OOF SVM margin 的 sigmoid 校准，每个校准训练折重新执行 MATLAB SFS |
| Habitat 终止 | 从深度 0 起比较开发训练内平均 CV AUC，在首次下降前停止；安全上限暂设 3，达到上限会记录 `depth_cap_reached` |
| K-means | 每轮每患者 10 类，10 次初始化；随机种子固定 |
| 输入体积 | ROI 包围盒重采样为 96³，不将包围盒内 ROI 外体素清零 |
| ResNet34 | MONAI 三维 ResNet34，宽度系数 1，无预训练权重 |
| Swin | MONAI 三维 Swin 编码器 + 全局池化 + 分类层；embed 24，depths [2,2,2,2]，heads [3,6,12,24]，window 7³，patch 2³；无预训练权重 |
| 训练 | AdamW，学习率 1e-4，weight decay 1e-4，batch 4，50 epochs，固定轮数；其余参数见配置 |
| 分类阈值 | 默认由当前训练集的完整流程 OOF 预测按 Youden 指数选择；也支持预先指定固定阈值 |
| 区间估计 | AUC 用 DeLong 正态近似区间并截至 [0,1]；比例指标用 Wilson 区间 |
| 比较与解释 | 外部配对双侧 DeLong，未作多重比较校正；DCA 为探索性；Kernel SHAP 的背景来自开发患者 |

MATLAB 的 `KernelScale = 1/sqrt(gamma)` 与 Python 的 `exp(-gamma * squared_distance)` 对应。两端在各训练折内各自估计缺失值中位数和标准化参数；最终预测来自 Python 拟合的 SVM。求解器不同，不能保证两端支持向量或小数末位完全相同。[MathWorks fitcsvm](https://www.mathworks.com/help/stats/fitcsvm.html)

Swin 的具体模块实现使用 MONAI 的三维 Swin 编码器，不能据此断言它与原研究未提供的网络代码逐层一致。[MONAI networks](https://monai.readthedocs.io/en/latest/networks.html) 影像组学参数的含义见 [PyRadiomics customization](https://pyradiomics.readthedocs.io/en/latest/customization.html)。

## 安装

使用 Python 3.11 和 MATLAB（本地验证版本为 R2024a）。MATLAB 需要 Statistics and Machine Learning Toolbox；Python 负责神经网络，因此不要求 MATLAB Deep Learning Toolbox。本机测试采用 Apple Silicon 原生 Python；其他平台应安装与硬件匹配的 PyTorch。

在解压后的项目目录执行：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install "setuptools<81" wheel "numpy==1.26.4"
python -m pip install -r requirements.txt --no-build-isolation
python -m pip install --no-deps --no-build-isolation -e .
```

Windows 激活环境使用 `.venv\Scripts\activate`。PyRadiomics 3.0.1 在部分平台需要本地 C/C++ 编译工具。本实现固定 Python 3.11，避免其旧构建脚本与 Python 3.12 的兼容问题。`requirements.tested.txt` 记录本次实际测试版本；CUDA 环境应另外核对 PyTorch 安装来源。

MATLAB 可执行文件自动从 PATH 或 macOS Applications 目录查找；其他位置在 `matlab.executable` 设置完整路径。两端通过本地 MAT 文件交换数据，不需要 MATLAB Engine for Python，不向外部服务器传送数据。

## 输入表

复制 `patients.example.csv`，每行一个患者：

| 列 | 内容 |
|---|---|
| patient_id | 全队列唯一字符串；同一患者不得以不同记录进入不同折 |
| center | A、B、C，与配置对应 |
| image | 原始 HU 三维影像文件路径，例如 `.nii.gz`；相对路径相对于 CSV 所在目录 |
| mask | 与图像尺寸、方向、原点、体素间距一致的二值 ROI；背景 0、病灶 1 |
| label | HT 为 1，CS 为 0；只做保存模型预测时可省略 |

同一患者的多个病灶应合并到同一患者 ROI。程序检查重复患者、重复图像路径、相同文件内容、几何不一致、空 ROI 与标签。队列数目严格核对为 299 / 125，不硬编码未确认的类别人数。人工分割一致性、参考标准与入排标准仍需要研究数据记录支持。

## 运行方式一：Python 入口

```bash
python -m hemorrhage validate --config config.study.json --manifest patients.csv --output runs/input_check
python -m hemorrhage run --config config.study.json --manifest patients.csv --output runs/study
```

运行时 Python 自动启动一个 MATLAB SFS 工作进程并持续复用，结束后关闭。默认同时生成 Python 与 MATLAB 统计结果。中断后只在配置、代码、数据和环境指纹一致时恢复：

```bash
python -m hemorrhage run --config config.study.json --manifest patients.csv --output runs/study --resume
```

## 运行方式二：MATLAB 入口

在 MATLAB 中，将路径替换为本机实际完整路径：

```matlab
addpath('/path/to/hemorrhage_reproduction_v01/matlab');
run_reproduction('/path/to/config.study.json', '/path/to/patients.csv', '/path/to/runs/study', '/path/to/.venv/bin/python');
```

此方式由当前 MATLAB 会话执行 SFS，同时启动 Python 子进程。计算期间查看输出目录中的 `python_execution.log`。恢复时加第 5 个参数 `true`。Windows 的 Python 路径通常为 `.venv\Scripts\python.exe`。

## 合成数据验证

```bash
python -m pytest tests -q
python -m hemorrhage demo --config config.study.json --output demo_data
python -m hemorrhage run --config demo_data/config.demo.json --manifest demo_data/patients.csv --output runs/demo
```

合成示例包含 64 名开发对象及 12 名外部对象，采用两折一次测试、缩小网络、1 epoch 和固定阈值以控制测试时间。它不满足正式研究的样本量或验证配置。该例的小 ROI 会触发无法形成 10 类的 habitat 情形，程序记录原因并采用开发阶段可用的先前深度；真实 patch 特征、10 类删除与多轮 OOF 链另经专门测试验证；另一个 16 例训练、2 例预测的合成例已完成实际 OOF habitat 生成、局部重训与冻结预测。不能将示例输出用作论文结果。

## 输出与复用

- `patient_folds.csv`：所有内部测试划分，四个模型共用患者折。
- `internal_predictions.csv`：各次重复、各折的测试概率与训练内选定阈值。
- `internal_metrics_by_repetition.csv`、`internal_descriptive_summary.csv`：每次重复的指标及均值、标准差。重复测试不被当作独立患者扩大样本量。
- `internal_patient_mean_predictions.csv`：每位患者多次测试概率的描述性汇总，不据此声称独立外部验证。
- `final_models`：全部开发患者重训后的模型、所选特征、SFS 路径、网络参数、阈值和深度选择记录。
- `external_predictions.csv`：冻结模型的外部预测；`evaluation` 与 `matlab_evaluation`：分类指标、DeLong、DCA、ROC / DCA 图。
- `external_habitats`：每位外部患者的概率图、逐轮保留掩膜和最终 habitat，目录以患者标识哈希命名；对应标识在 JSON 内。
- `shap`：最终 habitat 分类器的外部特征贡献表和图；它不是体素归因图，也不证明生物学机制。
- `fit_audit.jsonl`、`config.json`、`environment.json`、`run_state.json`：训练和评价患者范围、参数、版本、输入与源码指纹。
- `cache`：预处理图像、radiomics、patch 特征及 OOF masks，可复用以减少重复计算。

加载已拟合模型预测，不再启动 SFS：

```bash
python -m hemorrhage predict --model runs/study/final_models/habitat/model.joblib --manifest new_patients.csv --output runs/new_predictions
```

单独重新计算已有外部预测表：

```bash
python -m hemorrhage evaluate --config config.study.json --predictions runs/study/external_predictions.csv --output runs/re_evaluation
```

```matlab
evaluate_predictions('/path/to/external_predictions.csv', '/path/to/matlab_evaluation', '/path/to/config.study.json');
```

## 计算与解释边界

每一轮 OOF habitat 重新排除被预测患者及其对先前训练链的影响。加上 SFS、校准、深度选择、阈值选择和五折重复五次，正式运行会产生较多模型；20³、步长 1 的 patch 提取也可能占用大量时间和磁盘。默认顺序执行并缓存结果。应先运行合成验证和小规模技术检查，再配置有足够资源的正式实验。

偶数 patch 以当前 ROI 体素为锚点，范围从该坐标减 10 起；到达整幅扫描边界时移动窗口以保持 20 体素，扫描本身不足 20 体素时复制图像边缘补齐。这属于尚未从原始代码确认的扫描边界处理。ROI 外真实图像仍进入 patch。

训练集内部若概率图不足以形成 10 个非空类，程序记录当前深度不可用，不静默改为少于 10 类。冻结模型在外部患者上出现这种情形时会报错，不利用该患者的结局调整深度。出现训练子折类别人数不足时同样报错，不自动降低折数。正式数据应核查这些失败及其原因。

本实现保证外层测试与外部评价不参与训练、特征选择或参数选择。Habitat 的中间训练特征使用 OOF 图生成；SFS 和概率校准使用这些训练范围内已生成的特征。其内部 AUC 仅服务于开发选择，最终泛化性能应依据独立外层测试和外部验证评价。
