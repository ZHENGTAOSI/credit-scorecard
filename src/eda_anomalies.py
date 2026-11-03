"""EDA 第 2 步：三个异常值的分布细节（为处理决策提供依据）。"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("data/processed/cs-training-clean.csv", index_col=0)
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

# 1. age：分布 + 未成年行数
print("age 描述统计：")
print(df["age"].describe())
print(f"\nage < 18 的行数: {(df['age'] < 18).sum()}")

fig, ax = plt.subplots()
df.loc[df["SeriousDlqin2yrs"] == 0, "age"].plot.hist(
    bins=50, alpha=0.6, label="正常", ax=ax
)
df.loc[df["SeriousDlqin2yrs"] == 1, "age"].plot.hist(
    bins=50, alpha=0.6, label="违约", ax=ax
)
ax.legend()
fig.savefig(OUT / "age_by_target.png", dpi=150, bbox_inches="tight")

# 2. 逾期次数：96/98 占位符有多少
dpd_cols = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]
for col in dpd_cols:
    weird = df[col].isin([96, 98]).sum()
    print(f"{col}: 96/98 共 {weird} 行")

# 3. 额度使用率：超过 1 的有多少
util = df["RevolvingUtilizationOfUnsecuredLines"]
print(f"\n使用率 > 1 的行数: {(util > 1).sum()}（占比 {(util > 1).mean():.2%}）")
print(util.describe())