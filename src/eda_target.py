"""EDA 第 1 步：目标变量分布（脚本版，替代 notebook）。"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DATA_PATH = Path("data/processed/cs-training-clean.csv")
OUTPUT_DIR = Path("outputs")

df = pd.read_csv(DATA_PATH, index_col=0)
target = df["SeriousDlqin2yrs"]
print(f"违约率: {target.mean():.2%}")

counts = target.value_counts()
fig, ax = plt.subplots()
counts.plot(kind="bar", ax=ax)
ax.set_title("Target Distribution")
ax.set_xticklabels(["正常", "违约"], rotation=0)
ax.set_ylabel("样本数")

OUTPUT_DIR.mkdir(exist_ok=True)
fig.savefig(OUTPUT_DIR / "target_dist.png", dpi=150, bbox_inches="tight")
print(f"图已保存到 {OUTPUT_DIR / 'target_dist.png'}")