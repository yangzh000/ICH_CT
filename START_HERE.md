# 快速开始

MATLAB 负责 SFS，Python 负责其余建模。所有 .m 与 .py 文件均无注释。

1. 安装 Python 3.11、MATLAB 及 Statistics and Machine Learning Toolbox。
2. 在解压目录安装依赖：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install "setuptools<81" wheel "numpy==1.26.4"
python -m pip install -r requirements.txt --no-build-isolation
python -m pip install --no-deps --no-build-isolation -e .
```

3. 按 patients.example.csv 填写患者表，HT = 1、CS = 0。核对 config.study.json 中尚未确认的参数，具体见 README.md。
4. 启动：

```bash
python -m hemorrhage run --config config.study.json --manifest patients.csv --output runs/study
```

Python 会自动启动并复用 MATLAB 执行 SFS。也可从 MATLAB 启动：

```matlab
addpath('/path/to/hemorrhage_reproduction_v01/matlab');
run_reproduction('/path/to/config.study.json', '/path/to/patients.csv', '/path/to/runs/study', '/path/to/.venv/bin/python');
```

核心文件：matlab/sfs_select.m、hemorrhage/selection.py、hemorrhage/habitat.py、hemorrhage/networks.py。输出保存在运行目录，包括所选特征、模型、患者折、预测、统计表和图件。

已完成必要的合成数据运行核验，无须先重复测试即可配置正式数据。缺少原始患者数据及部分原始参数，尚不能保证复现文章中的具体指标。
