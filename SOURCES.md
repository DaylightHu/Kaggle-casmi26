# 参考与实现来源

本项目为独立编写的检索与可选指纹基线。没有复制参考笔记本的嵌入式引擎、融合权重、预训练参数或其数据资产，也没有声称复现其公开成绩。

- 用户提供的比赛和数据说明是规模、十种测试加合物、MRR@25、RDKit 版本、数据列、三类新颖性及截止日期的主要依据。
- 用户上传的 `casmi-26-from-spectra-to-structures.ipynb`：检查了输入处理、评分身份陷阱、外部依赖和验证讨论。原笔记本标识为 `dmitriigluzdov/casmi-26-eda-search-model`：<https://www.kaggle.com/code/dmitriigluzdov/casmi-26-eda-search-model>。其多引擎实现不包含在本项目中。
- 比赛：<https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra>。
- 数据说明：<https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/data>。
- 评价说明：<https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/overview/evaluation>。
- 用户提及的社区资源：<https://www.kaggle.com/datasets/daylighth/casmi26-community>。网页内容本次无法取得；项目提供显式字段导入，不对实际文件列表作假设。
- RDKit `TautomerEnumerator.Canonicalize` 官方文档：<https://www.rdkit.org/docs/source/rdkit.Chem.MolStandardize.rdMolStandardize.html>。在线文档可能对应比评分固定版本更新的 RDKit；代码运行时仍检查 `2026.03.3`。
- RDKit 2026.3.3 Python 分发：<https://pypi.org/project/rdkit/2026.3.3/>。
- PyTorch 2.7.1 / CUDA 11.8 安装指令：<https://pytorch.org/get-started/previous-versions/>。
- 公开结构资料可从 COCONUT、PubChem 等原始来源准备；本代码包不随附任何外部数据集或权重。各外部资源保持原有许可证，需分别检查。

项目代码采用 MIT 许可；该许可仅覆盖本项目新增源码，不覆盖比赛数据、社区资料或外部模型。
