# VQAv2 Inference with Recursive Tool Calling

A sophisticated VQA (Visual Question Answering) inference system for VQAv2 dataset that uses recursive decomposition and tool calling to answer complex visual questions.

## Features

- **Recursive Question Decomposition**: Automatically breaks down complex questions into simpler atomic sub-questions
- **Atomicity Checking**: Uses LLM to determine if a question can be answered directly or needs decomposition
- **Tool Integration**: Supports Grounding DINO for object detection and OCR for text recognition
- **Visual Context Extraction**: Uses CLIP to extract top-10 most likely visual concepts for context
- **Flexible LLM Backend**: Uses HuggingFace Transformers, compatible with any causal language model
- **Configurable Recursion Depth**: Prevents infinite recursion with configurable depth limits (default: 3)
- **Result Aggregation**: Intelligently combines answers from sub-questions into final answer

## Architecture

The system implements a recursive answer function with the following flow:

```
answer(image, question, depth=0)
    │
    ├─→ Extract visual context using CLIP (top-10 concepts)
    │
    ├─→ Check atomicity using LLM
    │
    ├─→ If ATOMIC:
    │   ├─→ Generate tool call using LLM
    │   └─→ Execute tool (Grounding DINO or OCR)
    │
    └─→ If NOT ATOMIC:
        ├─→ Generate sub-questions using LLM
        ├─→ Recursively call answer() for each sub-question
        └─→ Aggregate results using LLM
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/oguz-kolukisa/seq-r1-tool-calling.git
cd seq-r1-tool-calling
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) For actual Grounding DINO and OCR functionality, install additional dependencies:
```bash
# For Grounding DINO
pip install groundingdino

# For OCR (EasyOCR)
pip install easyocr
```

## Usage

### Command Line Interface

Run inference on a single image-question pair:

```bash
python run_inference.py \
    --image path/to/image.jpg \
    --question "How many people are in the image?" \
    --max-depth 3 \
    --output result.json
```

Options:
- `--image`: Path to input image (required)
- `--question`: Question about the image (required)
- `--llm-model`: HuggingFace LLM model name (optional, defaults to config)
- `--clip-model`: HuggingFace CLIP model name (optional, defaults to config)
- `--max-depth`: Maximum recursion depth (default: 3)
- `--output`: Path to save JSON output (optional)

### Python API

```python
from inference import VQAInference

# Initialize inference engine
vqa_engine = VQAInference(
    llm_model_name="meta-llama/Llama-2-7b-chat-hf",  # or any HF model
    clip_model_name="openai/clip-vit-base-patch32",
    max_depth=3
)

# Single inference
result = vqa_engine.inference_vqav2(
    image_path="path/to/image.jpg",
    question="What is happening in this image?"
)

print(f"Answer: {result['answer']}")

# Batch inference
data = [
    ("image1.jpg", "How many cars are visible?"),
    ("image2.jpg", "What color is the building?"),
]
results = vqa_engine.batch_inference(data)
```

### Direct Answer Function

```python
from PIL import Image
from inference import VQAInference

vqa_engine = VQAInference(max_depth=3)
image = Image.open("path/to/image.jpg").convert('RGB')

# Recursive answer function
answer = vqa_engine.answer(image, "Your question here", depth=0)
```

## Components

### 1. Configuration (`config.py`)
- Model names and settings
- Prompt templates for different functions
- Tool configurations
- Default parameters

### 2. CLIP Context Extractor (`clip_extractor.py`)
- Extracts top-10 most likely visual concepts from images
- Provides context to LLM (which doesn't see images directly)

### 3. LLM Wrapper (`llm_wrapper.py`)
- Wraps HuggingFace language models
- Implements four key functions:
  - `check_atomicity()`: Determines if question is atomic
  - `generate_tool_call()`: Generates tool call for atomic questions
  - `generate_sub_questions()`: Breaks down complex questions
  - `aggregate_results()`: Combines sub-question answers

### 4. Tools (`tools.py`)
- `GroundingDINOTool`: Object detection and grounding
- `OCRTool`: Text recognition
- `ToolExecutor`: Manages and executes tool calls

### 5. Main Inference Engine (`inference.py`)
- `VQAInference` class: Main orchestrator
- `answer()`: Recursive function implementing the core logic
- `inference_vqav2()`: High-level interface for single inference
- `batch_inference()`: Process multiple samples

## Configuration

Edit `config.py` to customize:

```python
# Default recursion depth
DEFAULT_DEPTH = 3

# Model configurations
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
LLM_MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"

# CLIP top-k tokens
CLIP_TOP_K_TOKENS = 10

# Prompt templates
ATOMICITY_CHECK_PROMPT = "..."
TOOL_CALL_GENERATION_PROMPT = "..."
SUB_QUESTION_GENERATION_PROMPT = "..."
AGGREGATE_RESULTS_PROMPT = "..."
```

## Examples

See `examples.py` for detailed usage examples:
- Single inference
- Batch inference
- Custom model configurations
- Direct answer function usage

## Tool Calling

The system supports two tools:

1. **Grounding DINO** (`grounding_dino`):
   ```python
   # For object detection and grounding tasks
   grounding_dino(query="person")
   grounding_dino(query="red car")
   ```

2. **OCR** (`ocr`):
   ```python
   # For text recognition tasks
   ocr()
   ```

The LLM generates appropriate tool calls based on the question type.

## Recursion Control

- Default maximum depth: 3 levels
- Prevents infinite recursion
- Configurable per inference call
- Depth tracking in verbose output

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- Pillow 9.0+
- CUDA (optional, for GPU acceleration)

See `requirements.txt` for complete list.

## Notes

- The current implementation uses placeholder tool implementations. For production use, integrate actual Grounding DINO and OCR models.
- LLM model requires appropriate HuggingFace access tokens for gated models (e.g., Llama-2)
- GPU is highly recommended for faster inference
- Adjust `max_length` and `temperature` parameters in `llm_wrapper.py` for different generation behaviors

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.