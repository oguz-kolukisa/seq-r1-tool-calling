# Quick Start Guide

This guide will help you get started with the VQAv2 Inference System quickly.

## Installation

1. **Clone the repository**:
```bash
git clone https://github.com/oguz-kolukisa/seq-r1-tool-calling.git
cd seq-r1-tool-calling
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Optional - Install actual tool implementations**:
```bash
# For Grounding DINO (object detection)
pip install groundingdino

# For OCR
pip install easyocr
```

## Basic Usage

### Command Line

Run inference on a single image-question pair:

```bash
python run_inference.py \
    --image path/to/your/image.jpg \
    --question "What is in this image?" \
    --max-depth 3
```

### Python Script

Create a simple script:

```python
from inference import VQAInference

# Initialize the VQA engine
vqa_engine = VQAInference(max_depth=3)

# Run inference
result = vqa_engine.inference_vqav2(
    image_path="path/to/image.jpg",
    question="How many people are in the image?"
)

print(f"Answer: {result['answer']}")
```

## Configuration

Edit `config.py` to change models or settings:

```python
# Use a different LLM model
LLM_MODEL_NAME = "microsoft/phi-2"  # Smaller, faster model

# Adjust recursion depth
DEFAULT_DEPTH = 2  # Less recursion, faster inference

# Change CLIP model
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"  # Larger model
```

## Understanding the System

### What happens when you ask a question?

1. **CLIP extracts visual context** - Gets the top 10 most likely visual concepts
2. **LLM checks if question is atomic** - Can it be answered with one tool call?
   - If **YES**: Generate and execute tool call (Grounding DINO or OCR)
   - If **NO**: Break into sub-questions and recursively process each
3. **Aggregate results** - Combine answers into final response

### Example Flow

**Question**: "How many red cars are visible?"

```
Recursion Depth 0:
  ├─ Context: "car, vehicle, outdoor, street, many, red, ..."
  ├─ Atomicity: NOT_ATOMIC (needs to find cars AND check color)
  └─ Sub-questions:
      ├─ "Where are the cars in the image?"
      └─ "Which cars are red?"

Recursion Depth 1:
  ├─ Sub-question 1: "Where are the cars?"
  │   ├─ Atomicity: ATOMIC
  │   └─ Tool: grounding_dino(query="car")
  │       → Result: "5 cars detected"
  │
  └─ Sub-question 2: "Which cars are red?"
      ├─ Atomicity: ATOMIC  
      └─ Tool: grounding_dino(query="red car")
          → Result: "3 red cars detected"

Back to Depth 0:
  └─ Aggregate: "There are 3 red cars visible in the image"
```

## Tips

### 1. Start with smaller models for testing
```python
# Fast, smaller model for development
vqa_engine = VQAInference(
    llm_model_name="microsoft/phi-2",
    max_depth=2
)
```

### 2. Use appropriate depth for your questions
- **depth=1**: Simple questions, one decomposition level
- **depth=2**: Moderate questions, good balance
- **depth=3**: Complex questions, most thorough (default)

### 3. Batch processing for efficiency
```python
data = [
    ("image1.jpg", "How many people?"),
    ("image2.jpg", "What color is the car?"),
    ("image3.jpg", "What does the sign say?"),
]

results = vqa_engine.batch_inference(data)
```

### 4. Check the logs
The system prints verbose logs showing:
- Recursion depth
- Atomicity decisions
- Tool calls
- Sub-questions generated
- Results at each level

## Common Issues

### 1. Out of Memory
**Problem**: GPU/CPU runs out of memory with large models

**Solution**:
- Use smaller models: `"microsoft/phi-2"` instead of `"meta-llama/Llama-2-7b-chat-hf"`
- Reduce batch size
- Use CPU instead of GPU for smaller models

### 2. Slow Inference
**Problem**: Takes too long to process questions

**Solution**:
- Reduce `max_depth` (2 instead of 3)
- Use smaller/faster models
- Enable GPU acceleration
- Use quantized models

### 3. Model Not Found
**Problem**: HuggingFace model cannot be downloaded

**Solution**:
- Check model name spelling
- Verify you have internet access
- For gated models (like Llama-2), you need HuggingFace token:
```python
from huggingface_hub import login
login(token="your_token_here")
```

### 4. Poor Quality Answers
**Problem**: Answers are not accurate

**Solution**:
- Use larger, more capable LLM models
- Adjust prompt templates in `config.py`
- Increase recursion depth
- Implement actual Grounding DINO and OCR tools (current are placeholders)

## Next Steps

1. **Read the full README.md** - Comprehensive documentation
2. **Check ARCHITECTURE.md** - Understand the system design
3. **See examples.py** - More usage examples
4. **Implement real tools** - Replace placeholder tools with actual Grounding DINO and OCR
5. **Customize prompts** - Edit `config.py` to improve performance

## Example Questions

### Simple (Atomic)
- "What color is the car?"
- "Is there text in the image?"
- "Where is the person?"

### Moderate (1-2 decomposition levels)
- "How many people are wearing hats?"
- "What is written on the sign?"
- "Are there more cars or motorcycles?"

### Complex (2-3 decomposition levels)
- "How many people are sitting at tables and what are they doing?"
- "What is the total number of red and blue objects?"
- "Describe the scene including people, objects, and text"

## Getting Help

- Check the README.md for detailed documentation
- Review ARCHITECTURE.md for system design
- Look at examples.py for more usage patterns
- Check the issue tracker on GitHub

## Performance Expectations

With GPU and default settings (Llama-2-7b, depth=3):
- Simple questions: 2-5 seconds
- Moderate questions: 5-15 seconds  
- Complex questions: 15-40 seconds

With smaller model (Phi-2) and depth=2:
- Simple questions: 1-2 seconds
- Moderate questions: 2-5 seconds
- Complex questions: 5-10 seconds
