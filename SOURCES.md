# 参考与实现来源

本项目为独立编写的检索、指纹与重排流水线。没有声称复现任何公开排行榜成绩。GLACIER 扩展显式使用下列公开冻结权重和其运行所需的固定上游源码；候选选择、预测去重、谱评分、聚合和融合代码属于本项目。

- 用户提供的比赛和数据说明是规模、十种测试加合物、MRR@25、RDKit 版本、数据列、三类新颖性及截止日期的主要依据。
- 用户上传的 `casmi-26-from-spectra-to-structures.ipynb`：检查了输入处理、评分身份陷阱、外部依赖和验证讨论。原笔记本标识为 `dmitriigluzdov/casmi-26-eda-search-model`：<https://www.kaggle.com/code/dmitriigluzdov/casmi-26-eda-search-model>。其多引擎实现不包含在本项目中。
- 比赛：<https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra>。
- 数据说明：<https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/data>。
- 评价说明：<https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/overview/evaluation>。
- 用户提及的社区资源：<https://www.kaggle.com/datasets/daylighth/casmi26-community>。网页内容本次无法取得；项目提供显式字段导入，不对实际文件列表作假设。
- 用户取得的 GLACIER 资源：<https://www.kaggle.com/datasets/ahmedberatozer/casmi26-glacier>。代码只加载其中 SHA256 固定的 `ckpt/glacier.pt`、`src/ms_pred/` 和 `shim/`，不导入其 `gl_runner.py` 或 `gl_fuse.py`。资源清单声明原始模型为 MassSpecGym GLACIER checkpoint，上游 `ms-pred` 固定在提交 `708148c2a8eb0c32120644436fefd2fe90cd2214`；上游仓库：<https://github.com/coleygroup/ms-pred>。外部文件仍适用各自许可证。
- RDKit `TautomerEnumerator.Canonicalize` 官方文档：<https://www.rdkit.org/docs/source/rdkit.Chem.MolStandardize.rdMolStandardize.html>。在线文档可能对应比评分固定版本更新的 RDKit；代码运行时仍检查 `2026.03.3`。
- RDKit 2026.3.3 Python 分发：<https://pypi.org/project/rdkit/2026.3.3/>。
- PyTorch 2.7.1 / CUDA 11.8 安装指令：<https://pytorch.org/get-started/previous-versions/>。
- 公开结构资料可从 COCONUT、PubChem 等原始来源准备；本代码包不随附任何外部数据集或权重。各外部资源保持原有许可证，需分别检查。

项目代码采用 MIT 许可；该许可仅覆盖本项目新增源码，不覆盖比赛数据、社区资料或外部模型。
