# Credit Scorecard（信用评分卡实践）

基于公开信贷数据（Kaggle: Give Me Some Credit）的信用评分卡全流程实现：
数据清洗 → 分箱与 WOE 编码 → IV 变量筛选 → 逻辑回归 → 标准评分卡转换 → KS / AUC / PSI 评估。

## 项目亮点

- 手写了分箱与 WOE/IV 计算，并与 scorecardpy 结果交叉验证
- 模块化代码结构（src/ 可复用，notebooks/ 仅做演示），非单一大 notebook
- 完整评估体系：KS、AUC、分箱坏账率、跨样本 PSI 稳定性检验
- 附带单元测试（tests/test_binning.py）与面试问答整理（docs/面试问答.md）

## 数据说明

- 数据来源：Kaggle "Give Me Some Credit" 竞赛数据集
- 下载方式：[https://www.kaggle.com/c/GiveMeSomeCredit]
- 目标变量：SeriousDlqin2yrs（两年内是否严重违约）
- 数据不入库，请自行下载后放入 `data/raw/`

## 快速开始

```bash
git clone https://github.com/ZHENGTAOSI/credit-scorecard.git
cd credit-scorecard
pip install -r requirements.txt
# 1. 下载数据集放入 data/raw/
python src/data_prep.py        # 数据清洗
python src/binning.py          # 分箱 + WOE + IV 筛选
python src/train.py            # 训练逻辑回归
python src/scorecard.py        # 转换评分卡
python src/evaluate.py         # 输出 KS / AUC / PSI


