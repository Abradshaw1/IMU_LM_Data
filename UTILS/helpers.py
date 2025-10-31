from __future__ import annotations
import re
from typing import Optional, Literal, List
import numpy as np
import pandas as pd

# Optional: requires scipy in requirements.txt
try:
    from scipy.signal import decimate
except Exception:
    decimate = None

def resample_df(
    df: pd.DataFrame, target_cols: List[str], factor: int = 2
) -> pd.DataFrame:
    """
    FIR low-pass + decimation downsample (e.g., 100 Hz → 50 Hz with factor=2).
    Assumes df is (roughly) uniformly sampled and target_cols are numeric.

    Keeps non-target columns by simple stride (iloc[::factor]) which is fine
    when labels/timestamps align with the decimated signal.
    """
    if decimate is None:
        raise ImportError("scipy is required: pip install scipy")

    # Downsample timestamp/labels/etc. by striding
    base = df.iloc[::factor].reset_index(drop=True)

    # Replace sensor columns with filtered+decimated versions
    for col in target_cols:
        base[col] = decimate(
            df[col].to_numpy(), q=factor, ftype="fir", zero_phase=True
        )
    return base

def zscore_normalize(arr: np.ndarray) -> np.ndarray:
    """Z-score normalization across columns (features)."""
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    std = np.where(std == 0, 1.0, std)
    return (arr - mean) / std

def convert_unit(
    arr: np.ndarray, kind: Optional[Literal["acc", "gyro"]] = None
) -> np.ndarray:
    """
    Convert IMU units:
    - 'acc': g → m/s² (× 9.8 to match your request)
    - 'gyro': deg/s → rad/s
    """
    if kind == "acc":
        return arr * 9.8
    if kind == "gyro":
        return arr * (np.pi / 180.0)
    return arr

def normalize_str(s: str) -> str:
    """Normalize arbitrary strings into snake_case alphanumerics."""
    s = s.strip()
    s = re.sub(r"(?<!^)(?=[A-Z])", "_", s)   # camelCase → snake_case
    s = re.sub(r"[\s\-]+", "_", s)           # spaces & hyphens → underscore
    s = re.sub(r"[^\w]", "", s)              # drop non-word chars
    return s.lower()

def norm_label(s: str) -> str:
    """Normalize labels consistently for BOTH raw data and mapping keys."""
    x = normalize_str(str(s))              # your existing helper
    x = re.sub(r'[_\s]+', '_', x)         # collapse multiple underscores/spaces -> single _
    x = x.strip('_')                       # trim leading/trailing _
    return x

def keyize(s: str) -> str:
    # trim, collapse internal whitespace, lowercase
    return " ".join(str(s).strip().split()).lower()

def _keyize(s: str) -> str:
    s = str(s)
    s = s.replace("\u00A0", " ")      # NBSP → space
    s = s.replace("\u2011", "-")      # non-breaking hyphen → hyphen
    s = s.strip().lower()
    s = s.replace("-", " ")           # hyphens → spaces
    s = re.sub(r"\s+", " ", s)        # collapse spaces
    return s
