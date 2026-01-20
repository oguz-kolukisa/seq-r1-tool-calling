# Setup and Installation Guide

This guide covers the complete setup process for the VQAv2 inference system with real Grounding DINO and OCR implementations.

## Quick Start

```bash
# 1. Clone repository
git clone https://github.com/oguz-kolukisa/seq-r1-tool-calling.git
cd seq-r1-tool-calling

# 2. Install dependencies
pip install -r requirements.txt

# 3. Setup Grounding DINO
python setup_grounding_dino.py
cd GroundingDINO
pip install -e .
cd ..

# 4. Download VQAv2 dataset (optional, ~13GB for train+val)
python download_vqav2.py

# 5. Run inference
python run_inference.py --image path/to/image.jpg --question "What is in this image?"
```

## Detailed Setup Instructions

### 1. System Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended for faster inference)
- ~15GB disk space for VQAv2 dataset
- ~2GB for model checkpoints

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- PyTorch (2.0+)
- Transformers (for Qwen 3B LLM and CLIP)
- EasyOCR (for text recognition)
- Grounding DINO (for object detection)
- Other utilities (tqdm, gdown, pycocotools, etc.)

### 3. Setup Grounding DINO

Grounding DINO requires additional setup:

```bash
# Run the automated setup script
python setup_grounding_dino.py
```

This will:
1. Clone the Grounding DINO repository
2. Download the model checkpoint (~600MB)
3. Verify the setup

Then install Grounding DINO:

```bash
cd GroundingDINO
pip install -e .
cd ..
```

**Manual Setup (if automated script fails):**

```bash
# Clone repository
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -e .
cd ..

# Download checkpoint
mkdir -p models
cd models
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ..
```

### 4. Download VQAv2 Dataset

The VQAv2 dataset includes questions, annotations, and COCO images:

```bash
python download_vqav2.py
```

This downloads:
- Train questions and annotations
- Validation questions and annotations
- COCO 2014 train images (~13GB)
- COCO 2014 val images (~6GB)

The script creates an index for easy access:
- `data/vqav2/train_index.json` - Training data index
- `data/vqav2/val_index.json` - Validation data index

**Skip dataset download:** If you just want to test the system, you can skip this step and use your own images.

### 5. Verify Installation

Run the validation script to ensure everything is set up correctly:

```bash
python validate_code.py
```

This checks:
- All modules import correctly
- Configuration is valid
- Tools are accessible
- Documentation is complete

## Model Configuration

The system uses the following models (configured in `config.py`):

### LLM: Qwen 3B
- **Model:** `Qwen/Qwen2.5-3B-Instruct`
- **Purpose:** Question decomposition, tool selection, answer aggregation
- **Size:** ~3GB
- **Auto-downloaded** on first use from HuggingFace

### CLIP: ViT-Base
- **Model:** `openai/clip-vit-base-patch32`
- **Purpose:** Visual context extraction (top-10 concepts)
- **Size:** ~600MB
- **Auto-downloaded** on first use from HuggingFace

### Grounding DINO
- **Model:** SwinT-OGC
- **Purpose:** Object detection and grounding
- **Size:** ~600MB
- **Requires manual setup** (see step 3)

### EasyOCR
- **Languages:** English
- **Purpose:** Text recognition
- **Size:** ~100MB
- **Auto-downloaded** on first use

## Usage Examples

### Basic Inference

```bash
python run_inference.py \
    --image path/to/image.jpg \
    --question "How many people are in the image?" \
    --max-depth 3
```

### Python API

```python
from inference import VQAInference

# Initialize with default models
vqa = VQAInference(max_depth=3)

# Run inference on a single image
result = vqa.inference_vqav2(
    image_path="path/to/image.jpg",
    question="What color is the car?"
)

print(f"Answer: {result['answer']}")
```

### Batch Processing on VQAv2 Dataset

```python
import json
from inference import VQAInference
from PIL import Image

# Load dataset index
with open('data/vqav2/val_index.json', 'r') as f:
    dataset = json.load(f)

# Initialize VQA system
vqa = VQAInference(max_depth=3)

# Process first 10 samples
for entry in dataset[:10]:
    result = vqa.inference_vqav2(
        image_path=entry['image_path'],
        question=entry['question']
    )
    
    print(f"Question: {entry['question']}")
    print(f"Predicted: {result['answer']}")
    if 'answers' in entry:
        print(f"Ground truth: {entry['answers'][:3]}")
    print()
```

### Custom Models

You can override the default models:

```python
from inference import VQAInference

vqa = VQAInference(
    llm_model_name="Qwen/Qwen2.5-7B-Instruct",  # Larger model
    clip_model_name="openai/clip-vit-large-patch14",  # Larger CLIP
    max_depth=2  # Less recursion
)
```

## Configuration

Edit `config.py` to customize:

```python
# Recursion depth
DEFAULT_DEPTH = 3  # Maximum recursion depth

# Models
LLM_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# Grounding DINO
GROUNDING_DINO_CONFIG = {
    "config_file": "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    "checkpoint": "models/groundingdino_swint_ogc.pth",
    "box_threshold": 0.35,  # Detection threshold
    "text_threshold": 0.25,  # Text matching threshold
}

# OCR
OCR_CONFIG = {
    "languages": ["en"],  # Add more languages as needed
    "gpu": True,  # Use GPU if available
}

# Dataset paths
VQAV2_CONFIG = {
    "data_dir": "data/vqav2",
    "coco_dir": "data/coco",
    "download_splits": ["train", "val"],
}
```

## Troubleshooting

### Grounding DINO Import Error

```
ModuleNotFoundError: No module named 'groundingdino'
```

**Solution:** Install Grounding DINO:
```bash
cd GroundingDINO
pip install -e .
cd ..
```

### CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. Use smaller model: Change LLM to 1B or 2B version
2. Reduce batch size: Process one image at a time
3. Use CPU: Set `device_map=None` in config

### EasyOCR Model Download Fails

**Solution:** Manually download models:
```bash
python -c "import easyocr; reader = easyocr.Reader(['en'])"
```

### VQAv2 Download Interrupted

**Solution:** Re-run the download script. It will skip already downloaded files:
```bash
python download_vqav2.py
```

## Performance

### Inference Speed (on GPU)

- Simple atomic question: 2-5 seconds
- 2-level recursion: 5-12 seconds
- 3-level recursion: 12-25 seconds

### Resource Usage

- GPU Memory: ~4-6GB (depends on model)
- CPU Memory: ~8GB
- Disk Space: ~20GB (with dataset)

## What's Implemented

✅ **Complete Implementation - No Placeholders**

### Core Components
- ✅ Recursive answer function with depth control
- ✅ Atomicity checking via Qwen 3B LLM
- ✅ Tool call generation
- ✅ Sub-question decomposition
- ✅ Result aggregation
- ✅ CLIP context extraction (top-10 visual concepts)

### Tools
- ✅ **Grounding DINO** - Full implementation with model loading and inference
- ✅ **EasyOCR** - Full implementation with text detection and recognition
- ✅ **CLIP** - Full implementation for visual context

### Models
- ✅ **Qwen 3B** - Complete LLM integration with chat template support
- ✅ **CLIP ViT-Base** - Visual understanding
- ✅ **Grounding DINO SwinT** - Object detection
- ✅ **EasyOCR** - Text recognition

### Dataset
- ✅ VQAv2 download script
- ✅ COCO images download
- ✅ Dataset indexing
- ✅ Easy data loading

### Scripts
- ✅ `download_vqav2.py` - Download and prepare VQAv2 dataset
- ✅ `setup_grounding_dino.py` - Setup Grounding DINO model
- ✅ `run_inference.py` - CLI for inference
- ✅ `examples.py` - Usage examples
- ✅ `validate_code.py` - System validation

## Known Limitations

1. **Grounding DINO setup** - Requires manual installation (automated script provided)
2. **GPU recommended** - CPU inference is slow
3. **English only** - OCR configured for English (can be changed in config)
4. **Model size** - Requires ~6GB GPU memory with default models

## Next Steps

1. **Evaluate on VQAv2:** Run evaluation on validation set
2. **Fine-tuning:** Fine-tune Qwen on VQAv2 for better performance
3. **Optimize prompts:** Improve prompts for better decomposition
4. **Add more tools:** Integrate additional vision tools
5. **Parallel processing:** Speed up sub-question processing

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the documentation in `ARCHITECTURE.md`
3. Open an issue on GitHub

## License

MIT License - See LICENSE file for details
