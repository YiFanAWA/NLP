# A Joint Entity Relation Extraction Method for Document Level Traditional Chinese Medicine Texts

##  Instructions
The code has been tested with Python 3. To install the dependencies, please run:
```
pip install -r requirements.txt
```

After downloading the datasets, please create a new folder `resources` and put the datasets into that folder.
Overall, the folder structure of the entire repo should look like:
```
...
models/
resources/
--- biorelex/
------- train.json
------- dev.json
scorer/
.gitignore

...
```
./TCMroberta
In this folder are the secondary pre-training language models in the field of TCM

For training, please refer to the scripts  `trainer.py`. For example, to train a basic model for TCMERE, you can simply run:
```
python trainer.py
```

There are some redundant code in this repo.Some of the code retains the source code names. I am going to remove them soon.

## 安装说明（示例）

下面给出三组常用的安装示例命令：**CPU-only**、**CUDA 11.7** 和 **CUDA 13.0**。根据你的操作系统与 CUDA 驱动选择合适的命令。推荐使用 conda 创建独立环境并安装 PyTorch（尤其在需要 GPU 时），DGL 在不同 CUDA 版本下有不同的 wheel 名称，请按需安装。

注意：以下命令以 PowerShell / Windows 为例；Linux 下把 `conda` 命令与环境路径按需调整。

1) CPU-only（推荐在没有 GPU 的开发机器上）

```powershell
conda create -n tcmere python=3.10 -y
conda activate tcmere
# 安装 pytorch CPU 版本（示例）
pip install torch --index-url https://download.pytorch.org/whl/cpu
# 安装其余依赖（不包含 dgl GPU 版）
pip install -r requirements.txt
# 若你不需要 DGL，可跳过 dgl 的安装；否则安装 CPU 版 DGL：
pip install dgl
```

2) CUDA 11.7（示例，适用于已安装 CUDA 11.7 驱动的机器）

```powershell
conda create -n tcmere-gpu python=3.10 -y
conda activate tcmere-gpu
# 使用官方 PyTorch wheel（请根据官方页面确认版本号），示例：
pip install torch==2.9.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
pip install transformers==4.57.1
# 安装其余依赖（不包含 dgl GPU 版）
pip install pyhocon boltons sqlitedict networkx
# 安装 DGL 对应 CUDA 版本（示例：CUDA11.7 -> dgl-cu117）
pip install dgl-cu117 -f https://data.dgl.ai/wheels/repo.html
```

3) CUDA 13.0（示例，适用于已安装 CUDA 13.0 驱动的机器）

```powershell
conda create -n tcmere-cu130 python=3.10 -y
conda activate tcmere-cu130
# 安装 PyTorch 与 CUDA13 对应的 wheel（示例）
pip install torch==2.9.1+cu130 -f https://download.pytorch.org/whl/torch_stable.html
pip install transformers==4.57.1
pip install pyhocon boltons sqlitedict networkx
# DGL 对应 CUDA13 可能为 dgl-cu130（以 DGL 官方 wheel 列表为准）
pip install dgl-cu130 -f https://data.dgl.ai/wheels/repo.html
```

其它说明：
- 如果你使用 `conda` 安装 PyTorch（推荐在 GPU 环境下），也可以使用 `conda install pytorch torchvision torchaudio -c pytorch` 并指定 `cudatoolkit` 版本；然后再用 pip 安装 `transformers` 与其它依赖。
- `requirements.txt` 中已经包含一些固定版本用于本地环境（如 `torch==2.9.1+cu130`、`transformers==4.57.1`、`dgl==2.0.0`）。若要在不同机器上复现，请优先根据机器的 CUDA 版本安装对应的 PyTorch 与 DGL，然后再安装其余库。
- 在 Windows 上启用 DGL GPU 支持常见问题较多，若出现 DGL 的 GPU 算子报错（如 "Device API cuda is not enabled"），建议在 Linux 环境中做 GPU+DGL 的训练。

