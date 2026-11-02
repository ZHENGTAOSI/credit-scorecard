"""数据清洗：读取原始数据，处理缺失值，输出清洗后的数据集。

缺失处理决策（2026-11-02）：
- MonthlyIncome 缺失率 19.8%，直接删行会损失 20% 样本且丢失最强预测变量之一，
  同时缺失非随机（不填收入的人往往风险更高），删行会引入系统性偏差。
  故采用中位数填充。
- NumberOfDependents 缺失率仅 2.6%，删行代价小，直接删除缺失行，
  避免引入人造值。
"""

from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/raw/cs-training.csv")
PROCESSED_PATH = Path("data/processed/cs-training-clean.csv")


def load_data(path: Path) -> pd.DataFrame:
    """读取原始训练数据。

    index_col=0 用于去掉 csv 中多余的行号列（读入后叫 Unnamed: 0）。
    """
    return pd.read_csv(path, index_col=0)


def missing_report(df: pd.DataFrame) -> pd.Series:
    """计算每列缺失率，按从高到低排序。"""
    return df.isna().mean().sort_values(ascending=False)


def fill_median(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """用中位数填充指定列的缺失值，返回新 DataFrame。"""
    median_value = df[column].median()
    df[column] = df[column].fillna(median_value)
    print(f"{column}: 用中位数 {median_value:.0f} 填充了缺失值")
    return df


def drop_missing_rows(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """删除指定列存在缺失值的行，返回新 DataFrame。"""
    before = len(df)
    df = df.dropna(subset=[column])
    print(f"{column}: 删除了 {before - len(df)} 行缺失记录")
    return df


def save_data(df: pd.DataFrame, path: Path) -> None:
    """保存清洗后的数据，自动创建目标目录。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    print(f"清洗后数据已保存到 {path}")


if __name__ == "__main__":
    df = load_data(RAW_PATH)
    print(f"原始数据形状: {df.shape}")

    df = fill_median(df, "MonthlyIncome")
    df = drop_missing_rows(df, "NumberOfDependents")

    print(f"\n清洗后数据形状: {df.shape}")
    print("\n清洗后缺失率检查（应全为 0）：")
    print(missing_report(df))

    save_data(df, PROCESSED_PATH)