# IMU Dataset Unification Pipeline

Data alignment and unification pipeline for wrist-worn IMU datasets.
This repository standardizes multiple wearable and activity-recognition datasets into a single unified schema suitable for downstream model training. The pipeline produces a **50 Hz, wrist-only IMU stream** with harmonized coordinate frames, consistent physical units, and a shared activity label taxonomy across all datasets.

## Overview

Each dataset is preprocessed individually before being merged into a unified representation that shares:
- **Common sensor channels**: accelerometer and gyroscope only
- **Standardized sampling rate**: 50 Hz continuous stream
- **Unified coordinate frame**: FLU (Forward-Left-Up) orientation
- **Consistent physical units**: acceleration in m/s² (gravity included), gyroscope in rad/s
- **Wrist-only placement**: single wrist sensor stream
- **Harmonized activity labels**: global activity ontology with preserved native labels
- **Canonical column names**: defined in the unification schema with strict ordering and data types

## Dataset Schema

**Primary Index**: `["dataset", "subject_id", "session_id", "timestamp_ns"]`

**Core Columns** (in order):
1. `dataset` — string
2. `subject_id` — string
3. `session_id` — string
4. `timestamp_ns` — int64 (nanoseconds, strictly non-decreasing)
5. `acc_x`, `acc_y`, `acc_z` — float32 (m/s², gravity included)
6. `gyro_x`, `gyro_y`, `gyro_z` — float32 (rad/s)
7. `global_activity_id` — int16 (9000 = unknown)
8. `global_activity_label` — string
9. `dataset_activity_id` — int16
10. `dataset_activity_label` — string

## Setup

### 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -U pip
pip install -r requirements.txt
```

## Usage

**Step 1** — Download the raw datasets and place them under `data/raw_data/`.

**Step 2** — Run each notebook under `Individual_dataloaders/` to clean, normalize, and export each dataset to `data/cleaned_premerge/`.

**Step 3** — Run `Unification/merge_pipeline.ipynb` to merge all cleaned datasets into a unified parquet file.

## Repository Structure

```
├── data/
│   ├── raw_data/              # Original downloaded datasets
│   ├── cleaned_premerge/      # Cleaned per-dataset Parquet files
│   └── merged_dataset/        # Final unified dataset
│
├── Individual_dataloaders/    # One subfolder per dataset
│   └── <DatasetName>/
│       └── load_and_preprocess.ipynb
│
├── RTdata_collection/         # Real-time data collection preprocessing
│   └── preprocess_files.ipynb
│
├── Unification/
│   ├── schemas/
│   │   ├── continuous_stream_schema.json
│   │   └── activity_mapping.json
│   └── merge_pipeline.ipynb
│
├── UTILS/
│   └── helpers.py
│
├── requirements.txt
└── README.md
```

## License

This project is intended for research and educational use.
