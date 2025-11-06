from __future__ import annotations
import re
from typing import Optional, Literal, List, Tuple, Dict
import numpy as np
import pandas as pd
from pathlib import Path
import json


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

## oppportunity-specific helpers

def _canon(s):
    if pd.isna(s): return ""
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


# def _canon_object(name: str) -> str:
#     s = normalize_str(name)
#     s = s.replace("door1", "door").replace("door2", "door")
#     s = s.replace("drawer1", "drawer").replace("drawer2", "drawer").replace("drawer3", "drawer")
#     s = s.replace("knife_cheese", "knife").replace("knife_salami", "knife")
#     s = s.replace("lazychair", "chair")
#     return s

# def make_verb_object(verb: str, obj: str) -> str:
#     v = normalize_str(verb)
#     o = _canon_object(obj)
#     if v == "drink" and o == "cup":
#         return "drink_from_cup"
#     if v == "sip" and o == "cup":
#         return "sip_cup"
#     return f"{v}_{o}"

def upsample_df_rate(df: pd.DataFrame, tcol: str, num_cols, src_hz: float, dst_hz: int) -> pd.DataFrame:
    """
    Resample to dst_hz using seconds, phase-locked to session start, and
    robust interpolation:
      - If >=2 points: linear interp with edge holding (np.interp default behavior).
      - If ==1 point: hold that single value across the grid.
      - If ==0 points: NaN (should not occur for acc_* here).
    """
    if df.empty:
        out = pd.DataFrame({tcol: np.array([], dtype=np.float64)})
        for c in num_cols: out[c] = np.nan
        return out

    # time axis in seconds (float64)
    t_src_full = pd.to_numeric(df[tcol], errors="coerce").to_numpy(dtype=np.float64)
    m_t = np.isfinite(t_src_full)
    if m_t.sum() == 0:
        out = pd.DataFrame({tcol: np.array([], dtype=np.float64)})
        for c in num_cols: out[c] = np.nan
        return out

    t_src = t_src_full[m_t]
    STEP_S = 1.0 / float(dst_hz)

    # phase-locked grid: round start to nearest 0.02s tick, floor end
    t0_round = np.round(t_src.min() / STEP_S) * STEP_S
    t1       = t_src.max()
    n_ticks  = int(np.floor((t1 - t0_round) / STEP_S)) + 1
    if n_ticks < 1:
        n_ticks = 1
    ticks = np.arange(n_ticks, dtype=np.int64)
    t_new = t0_round + ticks * STEP_S

    out = pd.DataFrame({tcol: t_new})

    # per-column robust interpolation
    for c in num_cols:
        v_full = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float64)
        m = m_t & np.isfinite(v_full)
        if m.sum() >= 2:
            # np.interp already holds edges; no NaNs
            out[c] = np.interp(t_new, t_src_full[m], v_full[m]).astype(np.float32)
        elif m.sum() == 1:
            # single finite point: hold constant
            val = float(v_full[m][0])
            out[c] = np.full(t_new.shape, val, dtype=np.float32)
        else:
            out[c] = np.nan  # truly missing channel
    return out


def read_opp_column_names(col_path: Path) -> List[str]:
    """
    'Column: 1 MILLISEC; ...' -> ['1 MILLISEC', '2 <NAME>', ...]
    Keep the numeric prefix; we'll strip it during canonicalization.
    """
    names = []
    for ln in col_path.read_text().splitlines():
        ln = ln.strip()
        if not ln or "Column:" not in ln:
            continue
        lhs = ln.split(";")[0].strip()
        names.append(lhs.replace("Column: ", ""))
    return names

_axis_fix = re.compile(r"(acc|gyro|magnetic)([xyz])$", re.IGNORECASE)

def canonicalize_opp_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Strip leading '<idx> ' prefix
    - Remove verbose prefixes
    - Lowercase + underscores
    - Ensure 'accx' -> 'acc_x' etc
    """
    df = df.copy()
    df.columns = [re.sub(r"^\d+\s+", "", c) for c in df.columns]
    df.columns = [
        c.replace("InertialMeasurementUnit ", "")
         .replace("Accelerometer ", "")
         .replace("_ ", " ")
        for c in df.columns
    ]
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    df.columns = [_axis_fix.sub(r"\1_\2", c) for c in df.columns]
    return df

def parse_opp_subject_session(stem: str) -> Tuple[str, str]:
    """
    Accepts: S1-ADL1, s2_adl3, S3-DRILL, S1-Drill, 1-ADL1, etc.
    Returns ('S<id>', session_upper)
    """
    m = re.match(r"^[sS]?(\d+)[-_](ADL\d+|DRILL)$", stem, flags=re.IGNORECASE)
    if not m:
        # last-chance soft parse: split on - or _ and pick tokens
        parts = re.split(r"[-_]", stem)
        if len(parts) >= 2 and parts[0].isdigit():
            subj, sess = parts[0], parts[1]
        elif len(parts) >= 2 and parts[0].lower().startswith("s") and parts[0][1:].isdigit():
            subj, sess = parts[0][1:], parts[1]
        else:
            raise ValueError(f"Unexpected Opportunity++ filename format: {stem}")
        return f"S{subj}", sess.upper()
    subj, sess = m.groups()
    return f"S{subj}", sess.upper()


def nearest_label_join_1d(
    src_ts_ns: np.ndarray,
    src_label_df: pd.DataFrame,
    target_ts_ns: np.ndarray,
    half_frame_ns: int,
) -> pd.DataFrame:
    """
    Align labels from (src_ts_ns, src_label_df) to target_ts_ns by nearest neighbor
    within a half-frame tolerance. If outside tolerance, forward-fill.
    """
    # Precondition
    order = np.argsort(src_ts_ns)
    src_ts_ns = src_ts_ns[order]
    src = src_label_df.iloc[order].reset_index(drop=True)

    # nearest neighbor
    idx = np.searchsorted(src_ts_ns, target_ts_ns, side="left")
    idx = np.clip(idx, 0, len(src_ts_ns) - 1)

    left_idx = np.maximum(idx - 1, 0)
    right_idx = idx

    # choose nearer of left/right
    left_dist = np.abs(target_ts_ns - src_ts_ns[left_idx])
    right_dist = np.abs(target_ts_ns - src_ts_ns[right_idx])
    choose_left = left_dist <= right_dist
    chosen = np.where(choose_left, left_idx, right_idx)

    # tolerance: where distance > half frame, we’ll ffill (later)
    dist = np.minimum(left_dist, right_dist)
    out = src.iloc[chosen].reset_index(drop=True).copy()
    out.loc[dist > half_frame_ns, :] = np.nan

    # forward-fill NaNs produced by tolerance gaps
    out = out.ffill().bfill()
    return out.reset_index(drop=True)



