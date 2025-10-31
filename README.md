# IMU_LM_Data

**Data alignment and unification pipeline for the IMU-LM foundation model.**  
This repository standardizes multiple wearable and activity-recognition datasets into a single unified schema suitable for downstream training of large-scale IMU foundation models.

## 🌐 Overview

The goal of this repository is to **consolidate heterogeneous IMU datasets** (e.g., RecoFit, PAMAP2, Opportunity++, Samosa, Wear, UT_Watch) into a **consistent, schema-aligned, and clean format**.  
Each dataset is preprocessed individually before being merged into a unified representation that shares:

- Common sensor channels (e.g., accelerometer, gyroscope, magnetometer)
- Consistent sampling rates and timestamps
- Harmonized activity labels and metadata
- Canonical column names defined in the unification schema

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
