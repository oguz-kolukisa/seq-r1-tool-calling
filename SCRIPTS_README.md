# Automated Setup Scripts

This directory contains automated bash scripts for easy setup and execution of VQAv2 inference and training.

## Scripts

### 1. `setup_and_run_inference.sh`
Automatically downloads VQAv2 dataset + COCO images (if not present) and runs inference.

**Usage:**
```bash
# Run with demo
./setup_and_run_inference.sh

# Run with custom image and question
./setup_and_run_inference.sh --image path/to/image.jpg --question "What is this?"

# With additional parameters
./setup_and_run_inference.sh --image img.jpg --question "How many people?" --max-depth 3
```

**What it does:**
1. Installs Python dependencies (if not installed)
2. Sets up Grounding DINO (if not setup)
3. Downloads VQAv2 dataset and COCO images (skips if already exists)
4. Runs inference with provided parameters or demo

### 2. `setup_and_run_training.sh`
Automatically downloads VQAv2 dataset + COCO images (if not present) and runs GRPO training.

**Usage:**
```bash
# Run training
./setup_and_run_training.sh
```

**What it does:**
1. Installs Python dependencies (if not installed)
2. Sets up Grounding DINO (if not setup)
3. Downloads VQAv2 dataset and COCO images (skips if already exists)
4. Runs GRPO training for sub-question generation

**Output:**
- Checkpoints: `checkpoints/subquestion_grpo/`
- Final model: `checkpoints/subquestion_grpo/final_model/`
- Training log: `training.log`
- Metrics: `logs/subquestion_grpo/`

**Monitor training:**
```bash
# View logs in real-time
tail -f training.log

# Filter by level
grep "ERROR" training.log
grep "DEBUG" training.log
```

## Smart Download Skipping

Both scripts use the updated `download_vqav2.py` which automatically:
- Checks if dataset files already exist
- Skips download if data is complete
- Verifies dataset integrity
- Shows statistics about existing data

This saves time and bandwidth on subsequent runs!

## Requirements

- Python 3.8+
- pip
- Internet connection (for first run to download datasets)
- ~20GB disk space for VQAv2 + COCO images

## Dataset Structure

After running either script, you'll have:
```
data/vqav2/
├── train/
│   ├── v2_OpenEnded_mscoco_train2014_questions.json
│   └── v2_mscoco_train2014_annotations.json
├── val/
│   ├── v2_OpenEnded_mscoco_val2014_questions.json
│   └── v2_mscoco_val2014_annotations.json
├── train_index.json
└── val_index.json

data/coco/
├── train2014/
│   └── COCO_train2014_*.jpg (82,783 images)
└── val2014/
    └── COCO_val2014_*.jpg (40,504 images)
```

## Troubleshooting

**Error: Permission denied**
```bash
chmod +x setup_and_run_inference.sh
chmod +x setup_and_run_training.sh
```

**Error: Module not found**
```bash
pip install -r requirements.txt
```

**Out of disk space**
The dataset requires ~20GB. Free up space or use a different location by modifying `config.py`:
```python
VQAV2_CONFIG = {
    "data_dir": "/path/to/your/vqav2",
    "coco_dir": "/path/to/your/coco",
}
```

**Download interrupted**
Simply run the script again - it will resume from where it left off.
