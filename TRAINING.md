# GRPO Training for Sub-Question Generation

This document explains how to train the LLM's sub-question generation function using Group Relative Policy Optimization (GRPO) with a judge LLM for scoring.

## Overview

The training script (`train_subquestion_grpo.py`) fine-tunes the LLM to generate better sub-questions for complex visual questions. It uses:

- **GRPO (Group Relative Policy Optimization)**: A reinforcement learning approach that generates multiple samples per question and uses relative rewards
- **Judge LLM**: A larger model that scores sub-questions across multiple criteria
- **Multi-Criteria Reward**: Evaluates sub-questions on 5 dimensions with configurable weights

## Architecture

### Components

1. **Policy Model**: The LLM being trained (default: Qwen 2.5-3B)
2. **Reference Model**: Frozen copy of the initial policy (for KL divergence)
3. **Judge LLM**: Larger model for scoring (default: Qwen 2.5-7B)

### Training Flow

```
For each training example:
1. Generate group_size (4) sub-question samples using policy model
2. Judge LLM scores each sample on 5 criteria
3. Compute advantages (normalized rewards across group)
4. Update policy model with GRPO loss
5. Add KL divergence penalty to stay close to reference model
```

## Reward Function

The judge LLM scores sub-questions on 5 criteria (1-10 scale each):

### 1. **Diversity** (Weight: 0.20)
- How different the sub-questions are from each other
- **1-3**: Very similar or repetitive questions
- **4-6**: Some variation but overlapping concepts
- **7-9**: Diverse questions covering different aspects
- **10**: Highly diverse, each explores unique angle

### 2. **Relevance** (Weight: 0.25)
- How relevant sub-questions are to the original question
- **1-3**: Off-topic or unrelated
- **4-6**: Partially relevant but miss key aspects
- **7-9**: Highly relevant, addressing core question
- **10**: Perfect relevance, directly supporting answer

### 3. **Answerability** (Weight: 0.25)
- Can sub-questions be answered independently with vision tools?
- **1-3**: Require complex reasoning or external knowledge
- **4-6**: Answerable but require multiple steps
- **7-9**: Directly answerable with available tools
- **10**: All questions atomic and perfectly answerable

### 4. **Completeness** (Weight: 0.20)
- Do sub-questions cover what's needed for the original question?
- **1-3**: Major gaps, missing critical information
- **4-6**: Partial coverage, some aspects missing
- **7-9**: Good coverage, most information present
- **10**: Complete coverage, answering all gives full answer

### 5. **Clarity** (Weight: 0.10)
- Are sub-questions clear and well-formed?
- **1-3**: Unclear, grammatically incorrect, or confusing
- **4-6**: Understandable but awkwardly phrased
- **7-9**: Clear and well-formed questions
- **10**: Perfectly clear, concise, and well-structured

**Total Reward**: Weighted sum normalized to 0-1 scale

## Usage

### Basic Training

```bash
python train_subquestion_grpo.py
```

### Custom Configuration

Modify the `TrainingConfig` class in the script:

```python
config = TrainingConfig(
    model_name="Qwen/Qwen2.5-3B-Instruct",
    judge_model_name="Qwen/Qwen2.5-7B-Instruct",
    learning_rate=1e-5,
    batch_size=4,
    num_epochs=3,
    group_size=4,  # Samples per question for GRPO
    kl_coef=0.1,   # KL divergence coefficient
    reward_weights={
        "diversity": 0.20,
        "relevance": 0.25,
        "answerability": 0.25,
        "completeness": 0.20,
        "clarity": 0.10,
    }
)
```

### Training Data Format

The script expects JSON data with format:

```json
[
  {
    "question": "How many people are near the red car?",
    "answer": "There are 2 people near the red car.",
    "context": "car, person, street, vehicle, red, outdoor",
    "image_id": "12345"
  },
  ...
]
```

**Fields**:
- `question` (required): The original complex question
- `answer` (optional): Ground truth answer for judging completeness
- `context` (optional): CLIP visual context (top-k concepts)
- `image_id` (optional): Image identifier

### Using VQAv2 Dataset

If you've downloaded VQAv2 using `download_vqav2.py`:

```python
dataset = load_training_data("data/vqav2/train_index.json", max_samples=1000)
```

### Using Custom Dataset

```python
dataset = load_training_data("path/to/your/dataset.json", max_samples=500)
```

## Training Process

### Step 1: Prepare Data

Either download VQAv2:
```bash
python download_vqav2.py
```

Or prepare your own JSON file with the format above.

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Run Training

```bash
python train_subquestion_grpo.py
```

### Training Progress

The script will show:
- Progress bar for each epoch
- Loss and reward metrics
- Periodic checkpoints
- Epoch summaries

Example output:
```
Epoch 1/3: 100%|████████| 200/200 [15:30<00:00, loss=0.4521, reward=0.654]

Epoch 1 Summary:
  Avg Loss: 0.4521
  Avg Reward: 0.654
  Max Reward: 0.892
```

## Output Files

### Checkpoints

Saved to `checkpoints/subquestion_grpo/`:
- `checkpoint_epoch{N}_step{M}.pt` - Periodic checkpoints
- `final_model/` - Final trained model directory

### Logs

Saved to `logs/subquestion_grpo/`:
- `epoch_{N}_metrics.json` - Detailed metrics for each epoch

### Using Trained Model

Load the trained model for inference:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "checkpoints/subquestion_grpo/final_model"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

# Use in your VQA system
from inference import VQAInference

vqa = VQAInference(max_depth=3, model_path=model_path)
```

## Hyperparameters

### Learning Rate
- **Default**: 1e-5
- **Range**: 1e-6 to 1e-4
- Lower for stable training, higher for faster convergence

### Group Size
- **Default**: 4
- **Range**: 2-8
- More samples = better advantage estimation, but slower

### KL Coefficient
- **Default**: 0.1
- **Range**: 0.01 to 0.5
- Controls how much the policy can deviate from reference model

### Temperature
- **Default**: 0.9
- **Range**: 0.7 to 1.2
- Higher = more diverse samples for training

### Reward Weights
Adjust based on your priorities:
- Increase **diversity** if sub-questions are too similar
- Increase **relevance** if sub-questions go off-topic
- Increase **answerability** if sub-questions are too abstract
- Increase **completeness** if sub-questions miss key information
- Increase **clarity** if sub-questions are poorly formed

## Advanced Usage

### Multi-GPU Training

The script automatically uses multiple GPUs if available:
```python
# Handled automatically via device_map="auto"
```

### Gradient Accumulation

For larger effective batch sizes:
```python
# Modify compute_grpo_loss to accumulate over multiple examples
```

### Custom Judge Model

Use a different judge model:
```python
config.judge_model_name = "meta-llama/Llama-3-8B-Instruct"
```

### LoRA/PEFT Integration

For parameter-efficient training (add to script):
```python
from peft import get_peft_model, LoraConfig

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1
)

self.policy_model = get_peft_model(self.policy_model, lora_config)
```

## Monitoring

### TensorBoard (Optional)

Add logging:
```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(config.log_dir)
writer.add_scalar('Loss/train', loss, step)
writer.add_scalar('Reward/mean', mean_reward, step)
```

View with:
```bash
tensorboard --logdir logs/subquestion_grpo
```

### Weights & Biases (Optional)

```python
import wandb

wandb.init(project="vqa-subquestion-grpo", config=vars(config))
wandb.log({"loss": loss, "reward": reward})
```

## Troubleshooting

### Out of Memory
- Reduce `batch_size` or `group_size`
- Use smaller models
- Enable gradient checkpointing
- Use FP16/BF16 training

### Low Rewards
- Check if questions are appropriate for decomposition
- Adjust reward weights
- Increase training epochs
- Use more training data

### Training Instability
- Reduce learning rate
- Increase KL coefficient
- Lower temperature
- Add gradient clipping (already included)

### Judge Model Too Slow
- Use smaller judge model
- Cache judge scores for common patterns
- Reduce group_size

## Example: Training on VQAv2

```python
# 1. Download dataset
python download_vqav2.py

# 2. Configure training
config = TrainingConfig(
    dataset_path="data/vqav2/train_index.json",
    num_epochs=5,
    learning_rate=5e-6,
    group_size=4
)

# 3. Load data (complex questions only)
dataset = load_training_data(config.dataset_path, max_samples=2000)

# 4. Train
trainer = SubQuestionGRPOTrainer(config)
trainer.train(dataset)

# 5. Evaluate
from inference import VQAInference
vqa = VQAInference(model_path="checkpoints/subquestion_grpo/final_model")
result = vqa.inference_vqav2("test_image.jpg", "Complex question?")
```

## Performance Tips

1. **Start Small**: Test with 100-200 examples first
2. **Monitor Rewards**: Track reward trends to tune weights
3. **Iterate Weights**: Adjust reward weights based on observed issues
4. **Use Ground Truth**: Provide ground truth answers when available
5. **Quality over Quantity**: Better to train on good examples than many poor ones

## Citation

If you use this training framework, please cite:

```bibtex
@software{vqa_grpo_training,
  title={GRPO Training for VQA Sub-Question Generation},
  author={Your Name},
  year={2024},
  url={https://github.com/oguz-kolukisa/seq-r1-tool-calling}
}
```

## License

Same as main project (see LICENSE file).
