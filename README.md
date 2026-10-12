# CASMI 2026：质量约束谱库检索 + 可选分子指纹模型

这是一个从零组织的比赛基线项目，可在本地 AutoDL 环境开发，再导出到 Kaggle 离线 Notebook 推理。默认数据路径为 `/root/autodl-tmp/data`。**代码包不包含比赛数据、外部数据库或已经训练的权重。只有显式执行 `train-fingerprint` 才会训练模型。**

## 1. 比赛要做什么

输入是未知小分子的串联质谱（MS/MS）：碎片质量 `ms2_mzs`、强度、前体质量、加合物、碰撞能量等。一种分子可以有多条谱。你需要按 **`molecule_id` 汇总全部谱图证据**，输出一个从最好到最差的结构候选列表，每个分子最多 25 个 SMILES。

提交格式为 `submission.csv`，只有 `molecule_id,smiles` 两列，候选之间用分号分隔，每个测试分子恰好一行。评价指标是 **MRR@25**：正确答案第一次出现在第 r 位得 `1/r`，前 25 位都没有得 0，再对分子取平均；分子谱图数量不改变其评价权重。

匹配先用 **RDKit 2026.03.3** 做互变异构体规范化，再比较 InChIKey 第一段。不同立体异构体、等价互变异构体不应占据多个候选位置。训练文件自带的 `inchikey14` 不一定已经经过这一步，本项目从 SMILES 重新计算评分结构键，用同一身份做候选去重和数据划分。

你提供的规则要求：Kaggle Notebook 提交、CPU/GPU 运行时间不超过 9 小时、关闭网络、输出文件命名 `submission.csv`；外部数据和预训练模型必须公开可用。当前公开的 `test.parquet` 是训练集中的格式样例，隐藏重跑会替换它。样例推理用于检查代码流程，不能作为隐藏测试泛化成绩。

根据你提供的时间表，入赛及组队截止 **2026-12-07 23:59 UTC**，最终提交截止 **2026-12-14 23:59 UTC**。报名、外部资料许可和最后时间表以比赛页面为准。

## 2. 本项目做到哪一步

| 模块 | 做法 | 是否需要训练 |
|---|---|---|
| 结构身份 | 官方版本 RDKit、互变异构体规范化、InChIKey14 去重 | 否 |
| 候选召回 | 前体质量 + 加合物 → 中性精确质量 → 质量窗口筛选结构 | 否 |
| 谱库评分 | 碎片 m/z 和中性丢失两路有容差余弦匹配；参考谱按模式、加合物、能量优先选择 | 否 |
| 多谱融合 | 每条查询谱找每个结构的最好参考谱；先按加合物/模式汇总，再跨组平均 | 否 |
| 外部结构库 | 导入 CSV/Parquet/SMILES，重算质量和指纹，按评分结构键去重 | 否 |
| 指纹模型 | 峰与中性丢失的加权 EmbeddingBag + 元数据 MLP → 2048 位 Morgan 指纹 | **是，可选** |
| 结构排序 | 谱库分数 + 可选预测指纹相似度 + 少量质量一致性先验 | 否 |
| 验证 | 整个评分结构身份留出；或天然产物 timsTOF 面板上的两种检索场景 | 否 |
| 提交检查 | ID 完整性、空值、重复、25 候选上限、SMILES 有效性及等价候选重复 | 否 |

**覆盖边界：**训练集结构 + 导入结构库形成候选池。第一类“有公开参考谱”的分子可做谱库检索；第二类“数据库中有结构、无公开谱”的分子，需要外部候选库和指纹模型，才能获得有信息的排序。第三类“PubChem 也没有的新结构”通常不在候选池，本项目**没有从头生成 SMILES 的模块**，候选池外的正确答案无法命中。它是方便验证、迭代和提交的检索基线，不是社区多模型方案的完整复现，也不保证排行榜分数。

## 3. 解压和安装（本地）

建议使用独立的 **Python 3.11 或 3.12** 环境；示例在 Python 3.12 下经过轻量验证。保留原来的数据目录。

```bash
cd /root/autodl-tmp
unzip casmi26-baseline.zip
cd casmi26-baseline

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps

python -m casmi_baseline.cli doctor
python -m casmi_baseline.cli inspect
```

`doctor` 会检查 RDKit、Arrow、Numba 等。基础检索无需 PyTorch、无需 GPU。Numba 用于加速峰匹配；缺失时能用 Python 回退，但处理全数据会慢很多。

若要使用可选模型，安装支持你的驱动及显卡的 PyTorch。RTX 2080 Ti 是 Turing，计算能力为 7.5；本项目只使用 FP32 / FP16 AMP，不使用 BF16 或 `torch.compile`。下面的安装命令取自 PyTorch 官方旧版安装页，使用 CUDA 11.8 wheel；驱动需要与 CUDA 11.8 运行时兼容。

```bash
nvidia-smi
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
python -m casmi_baseline.cli doctor
```

如服务器已有合适的 PyTorch，可直接保留它。CPU-only PyTorch 也可做指纹推理，但训练建议用 GPU。

## 4. 最快拿到一个提交：不训练

默认 `configs/local.yaml` 已填写你的数据路径。先确认里面有 `train.parquet`、`test.parquet` 和 `sample_submission.csv`。

```bash
# 一次性读训练集并生成参考谱、结构与指纹缓存；这是数据处理，不是模型训练。
python -m casmi_baseline.cli prepare

# 当前 test.parquet 上的纯检索推理。
python -m casmi_baseline.cli predict --output submission.csv

# 独立检查完整性和结构有效性。
python -m casmi_baseline.cli check-submission --path submission.csv
```

除提交文件外，还会生成 `submission.candidates.csv`（候选分项得分）和 `submission.diagnostics.json`（中性质量、谱数、质量窗口候选数、回退标记等）。提交时只使用 `submission.csv`。

也可覆盖数据和缓存路径；**全局参数写在子命令之前**：

```bash
python -m casmi_baseline.cli --data-dir /root/autodl-tmp/data --work-dir /root/autodl-tmp/casmi_work prepare
python -m casmi_baseline.cli --data-dir /root/autodl-tmp/data --work-dir /root/autodl-tmp/casmi_work predict --output submission.csv
```

缓存按行记录结构身份、前体、能量等，谱图存成磁盘映射的 `mzs.npy` / `weights.npy`，不会全量载入显存。默认每条谱保留至多 96 峰、强度阈值 0.1%、去掉前体附近及其上的峰，做平方根强度变换和 L2 归一化。**这些参数是起点，需要验证，不是经验最优值。**

完整缓存约 250 万条谱时，两张峰数组按 `250万 × 96 × (4+2) 字节` 估算约 **1.44 GB（十进制）**，另有元数据、结构表和打包指纹。建议预留至少 10–20 GB 空闲磁盘，尤其导出 Kaggle 资产时会复制缓存；系统内存从 16 GB 起更方便，但没有在你的机器上实测内存峰值或运行时间。RDKit 对约 27.5 万结构的规范化是预处理的重要开销，可以先在本地完成。

只想检查小规模读写时，用单独工作目录，避免留下不完整的正式缓存：

```bash
python -m casmi_baseline.cli --work-dir work_smoke prepare --max-rows 5000
python -m casmi_baseline.cli --work-dir work_smoke make-split --fraction 0.2 --molecules 50
python -m casmi_baseline.cli --work-dir work_smoke validate --scenario no-reference
```

部分缓存可用于开发验证，但 `predict` 和 `export-assets` 会拒绝用它生成正式提交资产。缓存目录非空时不会覆盖；若要改变峰数、强度阈值、指纹定义或 RDKit 版本，请选择新工作目录重新 `prepare`。

## 5. 验证：先看候选覆盖，再看排名

### A. 默认：身份留出，模拟无参考谱的已知结构

```bash
python -m casmi_baseline.cli make-split --kind identity --fraction 0.1 --molecules 400
python -m casmi_baseline.cli validate --scenario no-reference
```

约 10% 的评分结构身份被划为留出侧，从中最多选 400 个分子，每个最多 8 条不同加合物/能量的谱。验证时禁止检索留出身份在**所有库**里的谱。候选池仍包含这些结构：这模拟“结构已知但没有参考谱”，不是从头结构生成；这里的结构可用性是人为保留候选标签的假设，应与真实外部数据库覆盖率分开看。纯检索在这种场景可能很弱，是正常的。

训练指纹模型会排除整个留出集合，而不只是参与评分的那 400 个分子；同一个评分身份的其他 SMILES、互变异构体和跨库谱也一起排除。

输出包含：MRR@25、Top-1、Recall@25、`candidate_coverage`。最后一个是正确身份是否进入**实际打分候选列表**；如果为 0，排序器无法补救。可查看诊断中的质量回退或候选截断。

### B. 更贴近目标仪器：250 天然产物面板

用新的工作目录保存另一份 split 或先备份 `work/validation`；`make-split` 会重写这个目录中的 split、queries、truth。

```bash
python -m casmi_baseline.cli make-split --kind np-panel --molecules 250 --spectra-per-molecule 16

# 场景 1：查询来自 enveda-np-examples，屏蔽这一库的全部参考谱；其他库中的同一结构仍可检索。
python -m casmi_baseline.cli validate --scenario library

# 场景 2：同一结构在所有库里的参考谱都屏蔽；结构仍留在候选池。
python -m casmi_baseline.cli validate --scenario no-reference
```

这个面板都是常见天然产物，不能代表隐藏测试的新颖程度或类别比例。指纹模型仍须重新按这份 split 训练。程序拒绝拿全量训练模型或见过这些验证身份的权重报告验证成绩。

## 6. 可选训练：你执行下面的命令

这里提供的是指令。本项目创建时**没有执行训练**。推荐先完成第 5 节的一种划分，再训练 CV 模型：

```bash
# 默认 batch=256、FP16 AMP、12 epochs，每轮最多 1500 steps。
python -m casmi_baseline.cli train-fingerprint \
  --split work/validation/split.json \
  --device cuda --batch-size 256 --epochs 12 --steps-per-epoch 1500

# 用按验证 BCE 选出的权重，评估分子级真实排名。
python -m casmi_baseline.cli validate \
  --scenario no-reference --checkpoint work/models/cv/best.pt --device cuda
```

模型采用 0.1 Da 峰桶和中性丢失桶的加权 EmbeddingBag，配合加合物 embedding、前体/中性质量、能量及缺失标记、模式、仪器和峰数特征，预测 2048 位、半径 2 的 Morgan 指纹。训练按分子身份均衡抽样，随机取该分子的一条谱，并做峰 dropout；损失为 BCE + 0.2 × soft Dice。验证、推理从测量质量计算中性质量，不输入结构标签。

该模型是紧凑设计，默认配置约 720 万参数（实际数目随训练集加合物种类略变），适合在 11 GB 显存上尝试。**未在 RTX 2080 Ti 上实测，不能保证 batch=256 一定可用或训练耗时。**如显存不足，从 128 或 64 开始；写入 `history.json` 的 `cuda_peak_allocated_gb` 是张量峰值，不含驱动全部占用。可以先用 2 个 epoch 看损失趋势与用时，再决定正式训练预算，训练命令不会出现在 Kaggle 推理入口。

```bash
# 首次短训练用新输出目录，避免覆盖已有结果。
python -m casmi_baseline.cli train-fingerprint \
  --device cuda --batch-size 128 --epochs 2 --steps-per-epoch 500 \
  --output-dir work/models/trial

# 中断后恢复：保持原 epochs、batch、steps、split 和配置不变。
python -m casmi_baseline.cli train-fingerprint \
  --device cuda --batch-size 128 --epochs 2 --steps-per-epoch 500 \
  --output-dir work/models/trial --resume work/models/trial/last.pt
```

程序写入 `last.pt`、`best.pt`、`history.json`。`best.pt` 依据留出谱的 BCE 选择，**不等同于 MRR 最优**；应另用 `validate` 看排序结果。模型训练代码包含恢复状态，但断点恢复和 GPU 反向训练没有在本项目生成过程中执行。

确定方法及训练轮数后，再用全部结构训练正式权重：

```bash
# 把 epochs 改为你在验证中确定的轮数；全量训练无验证标签、不会做验证 early stopping。
python -m casmi_baseline.cli train-fingerprint \
  --full --device cuda --batch-size 256 --epochs 12 --steps-per-epoch 1500

python -m casmi_baseline.cli predict \
  --checkpoint work/models/full/final.pt --device cuda --output submission.csv
```

只用比赛结构池训练/检索仍然无法召回训练集中没有的结构。要拓展第二类分子的覆盖，继续下一节。

## 7. 使用社区数据或 COCONUT / PubChem 候选

你提供的社区地址是 <https://www.kaggle.com/datasets/daylighth/casmi26-community>。此次无法取得其实际文件清单，因此没有硬编码未核实的目录、文件名、指纹顺序或权重格式。项目不自动下载外部数据，也不假设社区 checkpoint 可以直接装入此模型。

先把公开候选资料下载到本地，查看实际字段：

```bash
python -m casmi_baseline.cli inspect-candidates --path /root/autodl-tmp/data/community
```

支持 CSV/TSV/Parquet 中的一列 SMILES，或每行一个 SMILES 的 `.smi` 文件；自动重算精确质量和本模型的指纹。下面的 `candidates.parquet` / `canonical_smiles` 是**示例占位，请替换为实际文件和列名**：

```bash
python -m casmi_baseline.cli import-candidates \
  --path /root/autodl-tmp/data/community/candidates.parquet \
  --smiles-column canonical_smiles --name community

# 可重复执行，追加其他结构库并去重。
python -m casmi_baseline.cli import-candidates \
  --path /root/autodl-tmp/data/coconut.csv --smiles-column smiles --name coconut

python -m casmi_baseline.cli predict \
  --checkpoint work/models/full/final.pt --device cuda --output submission.csv
```

外部文件中已有的质量、InChIKey 和指纹不直接当作可信匹配依据；本项目重新计算，保证与本地模型和评分身份一致。导入后的候选存于 `work/candidates`。CSV 重复 SMILES、立体异构体和等价互变异构体只留一个结构。外部 catalog 暂按表规模在 CPU 内存中管理，建议先从天然产物数据库或有质量索引的子库开始；不要直接把全部 PubChem 导入这个小基线。

只有结构、没有谱且没有指纹模型时，外部候选主要靠质量先验排序，对同分异构体没有足够识别能力。较合理的下一步是先导入 COCONUT 等天然产物结构库，再用指纹模型评估 `no-reference` 的排名。比赛合规性取决于你使用资源的公开性、发布时间和许可，见资源原始页面及比赛规则。

## 8. Kaggle 离线推理与提交

训练、缓存准备放在本地；提交 Notebook 只读缓存、候选和权重，对当前挂载的测试表推理。

### 8.1 导出推理资产

```bash
# 不带 --checkpoint 也可导出纯谱库基线。
python -m casmi_baseline.cli export-assets \
  --dest /root/autodl-tmp/casmi26-inference-assets \
  --checkpoint work/models/full/final.pt
```

导出目录包含源码、config、缓存、可选候选库、权重和 `infer.py`。把**目录内所有文件**上传为一个你自己的 Kaggle Dataset，目录层级必须保留。无需把原始训练数据复制到该数据集。导出只是复制文件，不训练、不提交到 Kaggle。

### 8.2 准备离线依赖

最终关闭网络的 Notebook 需要能导入 numpy/pandas/pyarrow/numba/PyYAML/tqdm，以及准确版本的 RDKit。可先在与你提交环境 Python 版本、Linux 架构匹配的联网准备环境中收集 wheel，再一起上传。下面假设在目标 Kaggle Python 环境或相同版本环境中执行：

```bash
python -m pip download -r requirements.txt --dest wheelhouse

# 用已有 wheelhouse 再导出一次到新的空目录。
python -m casmi_baseline.cli export-assets \
  --dest /root/autodl-tmp/casmi26-inference-assets-with-wheels \
  --checkpoint work/models/full/final.pt --wheelhouse wheelhouse
```

不要用 Python 3.11 的 RDKit wheel 在 Python 3.12 Kaggle 镜像上安装。GPU Notebook 通常已有 torch，但要在当前镜像检查；纯检索没有 torch 依赖。需要自行准备 torch 时请按目标环境准备合适 wheel，避免把本地 CUDA/CPython 不匹配的包混入。

### 8.3 Notebook 单元格

包内 `notebooks/kaggle_submission.ipynb` 是提交模板。设置 `ASSETS` 为你的资产目录；添加比赛数据和你的资产 Dataset；关闭 Internet。第一格只做离线 wheel 安装，第二格只启动推理。

```python
from pathlib import Path
import subprocess, sys

ASSETS = Path('/kaggle/input/your-casmi26-inference-assets')  # 改为实际目录
WHEELS = ASSETS / 'wheels'
if WHEELS.exists():
    subprocess.check_call([
        sys.executable, '-m', 'pip', 'install', '--no-index',
        '--find-links', str(WHEELS), '-r', str(ASSETS / 'requirements.txt')
    ])

subprocess.check_call([
    sys.executable, str(ASSETS / 'infer.py'), '--assets', str(ASSETS),
    '--output', '/kaggle/working/submission.csv'
])
```

资产不需要写入 `/kaggle/input`，Numba 编译缓存写到 `/kaggle/working`。每次重跑都从当前 `test.parquet` 推导 molecule ID；`sample_submission.csv` 只在 ID 完全一致时提供顺序，不使用缓存的样例答案。隐藏测试的 molecule ID 改变不会要求改代码。

先在同样禁网的环境提交保存版本检查时长、错误和输出文件，再使用 Kaggle 的 Submit。**本项目未运行比赛全量推理，未验证 9 小时限制，未提交任何比赛成绩。**默认每个结构每条查询最多比较 24 条参考谱，质量候选最多 5000 个，是控制开销的起点；截断会损失候选覆盖，诊断文件会记录。

## 9. 读代码及迭代

| 路径 | 用途 |
|---|---|
| `configs/local.yaml` | 路径及默认参数 |
| `casmi_baseline/chemistry.py` | 加合物质量、评分身份、Morgan 指纹 |
| `casmi_baseline/spectra.py` | 谱清洗、峰及中性丢失匹配 |
| `casmi_baseline/cache.py` | 流式训练集缓存 |
| `casmi_baseline/candidates.py` | 外部结构导入和质量召回 |
| `casmi_baseline/retrieval.py` | 多谱聚合、排序和提交 |
| `casmi_baseline/fingerprint.py` | 可选模型、训练、恢复及推理 |
| `casmi_baseline/validation.py` | 评分身份留出、天然产物面板 |
| `casmi_baseline/metric.py` | MRR@25 和提交检查 |
| `casmi_baseline/cli.py` | 命令行入口 |
| `kaggle/infer.py` | 离线推理入口 |
| `notebooks/` | 本地运行说明、Kaggle 提交模板 |
| `tests/` | 人工数据检查，无训练 |

优先迭代顺序：

1. 确认缓存/提交能跑通，检查质量分散、无候选回退、谱图清洗为空的比例。
2. 导入天然产物候选，比较覆盖率。
3. 用 `identity` / `np-panel` 的 `no-reference` 训练与检验指纹排序。
4. 在 `np-panel` 的 `library` 场景调质量窗、碎片容差、参考谱数量和融合权重。
5. 用独立留出集合检查稳定性，避免反复调同一面板后只报告最优结果。
6. 若要提高第三类分子的成绩，再引入公式预测、受约束结构生成或候选编辑；应另建生成有效率、质量/公式一致性及真实结构召回评价。

若质量窗口没有候选，程序先扩大窗口；仍无候选时返回最近质量结构以避免空提交，并记录 `nearest_outside_window`。这只是格式兜底，**不是新结构发现**，应优先扩大结构库覆盖或检查加合物换算。

## 10. 测试与实际验证范围

```bash
python -m pip install pytest
python -m pytest -q
```

测试使用人工构造的 Parquet 文件，检查十种测试加合物、额外常见电荷/多聚体换算、互变异构体与立体去重、谱清洗、质量窗口检索、多谱提交、结构留出、天然产物面板参考屏蔽、外部导入、隐藏 ID 变化、MRR 和错误提交拒绝。PyTorch 已安装时还检查前向输出与 checkpoint 推理读取；不执行 `optimizer.step()`。

轻量测试结果与环境记录见 `VALIDATION.md`。它们证明代码路径和约束处理，不说明比赛准确率。没有运行你的实际数据、训练、GPU 反向传播、完整规模性能评测或 Kaggle 提交。
