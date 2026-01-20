# VQAv2 Recursive Inference System

A VQA (Visual Question Answering) system that recursively decomposes complex visual questions and uses tools (Grounding DINO, OCR) to answer them.

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Setup Grounding DINO (optional)
python setup_grounding_dino.py
cd GroundingDINO && pip install -e . && cd ..

# Run
python run_inference.py --image image.jpg --question "What is this?"
```

## Core Concept

```
question → CLIP context → atomicity check
    ├─ ATOMIC: tool call → execute → LLM reasons → answer
    └─ NOT ATOMIC: sub-questions → recurse → LLM aggregates → answer
```

## Usage

**Python API:**
```python
from inference import VQAInference

vqa = VQAInference(max_depth=3)
result = vqa.inference_vqav2("image.jpg", "What color is the car?")
print(result['answer'])
```

**CLI:**
```bash
python run_inference.py --image img.jpg --question "How many people?" --max-depth 3
```

## Configuration

Edit `config.py`:
- `LLM_MODEL_NAME`: Qwen/Qwen2.5-3B-Instruct
- `CLIP_MODEL_NAME`: openai/clip-vit-base-patch32
- `DEFAULT_DEPTH`: 3 (max recursion depth)

## Dataset

Download VQAv2 dataset with COCO images:
```bash
python download_vqav2.py
```

## Files

**Core:**
- `inference.py` - Main VQA class with recursive answer()
- `llm_wrapper.py` - LLM for reasoning (Qwen 3B)
- `clip_extractor.py` - Visual context extraction
- `tools.py` - Grounding DINO & OCR implementations
- `config.py` - Configuration & prompts

**Utilities:**
- `run_inference.py` - CLI interface
- `download_vqav2.py` - Dataset downloader
- `setup_grounding_dino.py` - Model setup
- `test.py` - Tests

## Requirements

- Python 3.8+
- PyTorch 2.0+
- transformers, easyocr, pillow

See `requirements.txt` for full list.

## License

MIT License