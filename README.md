# IMU_LM_Data
**Data alignment and unification pipeline for the IMU-LM foundation model.**  
This repository standardizes multiple wearable and activity-recognition datasets into a single unified schema suitable for downstream training of large-scale IMU foundation models. The pipeline produces a **50 Hz, wrist-only IMU stream** with harmonized coordinate frames, consistent physical units, and a shared activity label taxonomy across all datasets.

## 🌐 Overview
The goal of this repository is to **consolidate heterogeneous IMU datasets** (e.g., RecoFit, PAMAP2, Opportunity++, Samosa, Wear, UT_Watch) into a **consistent, schema-aligned, and clean format**.  

Each dataset is preprocessed individually before being merged into a unified representation that shares:
- **Common sensor channels**: accelerometer and gyroscope only (magnetometer excluded)
- **Standardized sampling rate**: 50 Hz continuous stream (downsampled from higher rates where necessary)
- **Unified coordinate frame**: FLU (Forward–Left–Up) orientation for all axes
- **Consistent physical units**: acceleration in m/s² (gravity included), gyroscope in rad/s
- **Wrist-only placement**: single wrist sensor stream (chest/ankle sensors dropped in multi-device datasets)
- **Harmonized activity labels**: global activity ontology with preserved native labels for traceability
- **Canonical column names**: defined in the unification schema with strict ordering and data types

## 📊 Final Dataset Structure

### Target Schema
The unified dataset follows a strict schema optimized for time-series foundation model training:

**Primary Index**: `["dataset", "subject_id", "session_id", "timestamp_ns"]`

**Core Columns** (in order):
1. `dataset` — string (dataset identifier, e.g., "pamap2", "recofit")
2. `subject_id` — string (dataset-local participant ID)
3. `session_id` — string (recording session or trial identifier)
4. `timestamp_ns` — int64 (nanoseconds since session start; strictly non-decreasing)
5. `acc_x`, `acc_y`, `acc_z` — float32 (linear acceleration in m/s², gravity included)
6. `gyro_x`, `gyro_y`, `gyro_z` — float32 (angular velocity in rad/s)
7. `global_activity_id` — int16 (mapped to shared ontology; 9000 = unknown/other)
8. `global_activity_label` — string (human-readable activity name from ontology)
9. `dataset_activity_id` — int16 (original numeric activity ID from source dataset)
10. `dataset_activity_label` — string (original verbatim activity label for traceability)

### Normalization Specifications

**Sampling Rate**: All signals are resampled to exactly **50 Hz** using FIR filtering followed by decimation to prevent aliasing.

**Coordinate Frame Alignment**: All IMU axes are transformed to the **FLU (Forward–Left–Up)** coordinate system:
- **Forward**: direction of wrist extension (toward fingers)
- **Left**: perpendicular to Forward, pointing left (radial direction)
- **Up**: perpendicular to both, pointing away from the ground when arm hangs naturally

Datasets with varying native coordinate systems (e.g., NED, ENU, device-specific frames) undergo rotation matrix transformations to achieve FLU alignment.

**Physical Units**:
- **Accelerometer**: m/s² with gravity **included** (raw accelerometer reading; no gravity subtraction)
- **Gyroscope**: rad/s (converted from deg/s where necessary)

**Sensor Placement**: Only **wrist-worn sensors** are included in the final stream. For multi-device datasets (e.g., chest, ankle, wrist), only the wrist stream is retained.

### Activity Label Harmonization

The pipeline employs a **two-tier labeling system**:

1. **Global Ontology**: Activities are mapped to a shared taxonomy (e.g., `walk`, `run`, `adl_household_general`, `exercise_jump_rope`) with numeric IDs. Examples:
   - `walking`, `nordic_walking` → `walk (id=2)`
   - `vacuum_cleaning`, `ironing`, `folding_laundry` → `adl_household_general (id=13)`
   - `rope_jumping` → `exercise_jump_rope (id=10)`

2. **Native Label Preservation**: Original dataset labels and IDs are retained in `dataset_activity_id` and `dataset_activity_label` for full traceability and dataset-specific analysis.

**Unknown/Ambiguous Activities**: Any activity that cannot be confidently mapped to the global ontology (including transient states, unlabeled segments, or highly dataset-specific activities) receives `global_activity_id = 9000` with label `unknown_activity`.

**Label Mapping File**: The global ontology and all dataset-specific mappings are defined in `schemas/activity_mapping.json`, which serves as the authoritative source for label unification.


## ⚙️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/Abradshaw1/IMU_LM_Data.git
cd IMU_LM_Data
```

### 2. Create a virtual environment

```bash
python3 -m venv .IMUDATA
source .IMUDATA/bin/activate   # Windows: .IMUDATA\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -U pip
pip install -r requirements.txt
```

## 🚀 Usage

Step 1 — Dataset preprocessing

Run each notebook under `Individual_dataloaders/` to:

- Load raw dataset files
- Apply cleaning, filtering, and normalization
- Export the cleaned dataset to `data/cleaned_premerge/` as a Parquet file

Step 2 — Schema unification

In `Unification/merge_pipeline.ipynb`:

- Validate each dataset against `schemas/main_schema.json`
- Apply the global `schemas/activity_mapping.json` to harmonize activity labels
- Merge all datasets into `data/merged_dataset/unified_dataset.parquet`

Step 3 — Model integration

The unified dataset can be used directly by IMU foundation-model pipelines (e.g., transformer-based pretraining, contrastive learning, or self-supervised sequence models).

## 🧱 Repository Structure

```
IMU_LM_Data/
│
├── data/
│   ├── raw_data/              # Original downloaded datasets
│   ├── cleaned_premerge/      # Cleaned & standardized per-dataset Parquet files
│   └── merged_dataset/        # Final unified dataset for model training
│
├── Individual_dataloaders/
│   ├── Opportunity++/
│   ├── PAMAP2/
│   ├── RecoFit/
│   ├── Samosa/
│   ├── UT_Watch/
│   └── Wear/
│       ├── load_and_preprocess.ipynb
│       └── README.md
│   # Each subfolder: dataset-specific preprocessing, cleaning, mappings
│
├── Unification/
│   ├── schemas/
│   │   ├── main_schema.json         # Canonical column definitions (dtype, semantics)
│   │   └── activity_mapping.json    # Global label harmonization map
│   ├── merge_pipeline.ipynb        # Merges all cleaned datasets → unified Parquet
│   └── README.md                   # Explains schema design and merge rules
│
├── UTILS/
│   └── helpers.py                   # Common functions: schema validation, IO, mapping
│
├── requirements.txt                 # Core dependencies (numpy, pandas, pyarrow, etc.)
└── README.md                        # You are here
```

## 🧠 Future Extensions

- Add temporal resampling and device-alignment utilities
- Integrate automatic label cleaning and activity taxonomy matching
- Extend schema to multimodal sensor streams (e.g., PPG, ECG, audio)

## 📄 License

This project is intended for research and educational use within the IMU-LM project.
