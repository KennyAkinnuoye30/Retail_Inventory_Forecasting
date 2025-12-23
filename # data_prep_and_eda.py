# data_prep_and_eda.py
# Cleaning + quick EDA for Retail Inventory dataset

from pathlib import Path
import json
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# 0) Paths anchored to THIS file
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUT_DIR = BASE_DIR / "outputs"

RAW_PATH = RAW_DIR / "retail_store_inventory.csv"   # <- your raw filename here
PROCESSED_PATH = PROCESSED_DIR / "clean.csv"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("[PATH CHECK]")
print("  BASE_DIR      :", BASE_DIR)
print("  RAW_PATH      :", RAW_PATH)
print("  PROCESSED_PATH:", PROCESSED_PATH)
print("  OUT_DIR       :", OUT_DIR)

# ----------------------------
# 1) Dataset-specific settings
# ----------------------------
DATE_COL   = "Date"
TARGET_COL = "Demand Forecast"
ID_COLS    = ["Store ID", "Product ID"]

# ----------------------------
# 2) Helpers
# ----------------------------
def winsorize_iqr(df: pd.DataFrame, exclude: list[str] = []):
    """Cap outliers using IQR rule; skip any columns in 'exclude'."""
    num_cols = [c for c in df.select_dtypes(include=np.number).columns if c not in exclude]
    for c in num_cols:
        q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        iqr = q3 - q1
        if not np.isfinite(iqr) or iqr == 0:
            continue
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        df[c] = df[c].clip(lower=low, upper=high)
    return df

_num_like_re = re.compile(r"^\s*[-+]?[\d.,]+(?:[%$])?\s*$")

def coerce_numeric_safe(s: pd.Series) -> pd.Series:
    """
    Convert object series to numeric when it 'looks' numeric.
    Avoids pandas FutureWarning about errors='ignore' and
    prevents accidental conversion of true categorical text.
    """
    if s.dtype != "object":
        return s
    # Only attempt if the vast majority of non-null values look numeric
    mask = s.dropna().astype(str).str.match(_num_like_re)
    if len(mask) == 0 or mask.mean() < 0.8:
        return s
    # strip currency/percent/comma then coerce
    cleaned = s.astype(str).str.replace(r"[,$%]", "", regex=True)
    out = pd.to_numeric(cleaned, errors="coerce")
    # if conversion nuked almost everything, revert
    if out.notna().mean() < 0.8:
        return s
    return out

# ----------------------------
# 3) Main
# ----------------------------
def main():
    # Guard: make sure raw file exists
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"RAW_PATH not found: {RAW_PATH}\n"
            f"Contents of {RAW_DIR}:\n{list(RAW_DIR.glob('*'))}"
        )

    # 1) Load
    df = pd.read_csv(RAW_PATH)
    print(f"[load] shape={df.shape}")

    # 2) Basic hygiene
    before = len(df)
    df = df.drop_duplicates()
    print(f"[clean] dropped duplicates: {before - len(df)}")

    # strip whitespace in text columns
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()

    # 3) Types
    # Date -> datetime
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")

    # Numeric-looking objects -> numeric (safe)
    for c in df.columns:
        df[c] = coerce_numeric_safe(df[c])

    # 4) Missing values
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            if df[c].isna().any():
                df[c] = df[c].fillna(df[c].median())
        elif c == DATE_COL:
            # drop rows where dates failed to parse
            df = df[df[DATE_COL].notna()]
        else:
            if df[c].isna().any():
                df[c] = df[c].fillna("unknown")

    # 5) Optional outlier capping (skip target)
    df = winsorize_iqr(df, exclude=[TARGET_COL])

    # 6) Simple time features
    df["year"]       = df[DATE_COL].dt.year
    df["month"]      = df[DATE_COL].dt.month
    df["dayofweek"]  = df[DATE_COL].dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # 7) Save processed
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"[save] -> {PROCESSED_PATH.resolve()} rows={len(df)} cols={len(df.columns)}")
    print("[exists?]", PROCESSED_PATH.exists())

    # 8) Quick EDA artifacts
    summary = {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "nulls": df.isna().sum().to_dict()
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "eda_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[eda] wrote {OUT_DIR / 'eda_summary.json'}")

    # Basic plots for the target
    if TARGET_COL in df.columns:
        (OUT_DIR).mkdir(exist_ok=True)
        plt.figure()
        df[TARGET_COL].hist(bins=40)
        plt.title(f"Distribution of {TARGET_COL}")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"hist_{TARGET_COL}.png")
        plt.close()

        plt.figure()
        df.sort_values(DATE_COL).set_index(DATE_COL)[TARGET_COL].plot()
        plt.title(f"{TARGET_COL} over time")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"time_{TARGET_COL}.png")
        plt.close()

if __name__ == "__main__":
    main()
