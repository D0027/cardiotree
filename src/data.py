"""Data loading, cleaning and splitting."""
from __future__ import annotations
import pandas as pd
from sklearn.model_selection import train_test_split
from .config import DATA_PATH, RS, TEST_SIZE


def load_raw(path=DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    return df


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop duplicate rows (they leak between train/test for deep trees)."""
    n_dup = int(df.duplicated().sum())
    return df.drop_duplicates().reset_index(drop=True), n_dup


def load_clean() -> tuple[pd.DataFrame, int]:
    return clean(load_raw())


def split(df: pd.DataFrame):
    features = [c for c in df.columns if c != "target"]
    X, y = df[features], df["target"].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RS)
    return X, y, X_tr, X_te, y_tr, y_te, features
