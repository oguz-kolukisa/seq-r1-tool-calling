"""
GRPO Training Script for Sub-Question Generation

This script trains the LLM's sub-question generation function using
Group Relative Policy Optimization (GRPO) with a judge LLM for scoring.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Tuple
import json
import os
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np
import config


@dataclass
class TrainingConfig:
    """Configuration for GRPO training."""
    
    # Model paths
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"  # Model to train
    judge_model_name: str = "Qwen/Qwen2.5-7B-Instruct"  # Larger model for judging
    
    # Training hyperparameters
    learning_rate: float = 1e-5
    batch_size: int = 4
    num_epochs: int = 3
    group_size: int = 4  # Number of generations per question for GRPO
    kl_coef: float = 0.1  # KL divergence coefficient
    clip_range: float = 0.2  # PPO-style clipping range
    
    # Generation parameters
    max_new_tokens: int = 256
    temperature: float = 0.9
    top_p: float = 0.95
    
    # Reward weights for different criteria (sum to 1.0)
    reward_weights: Dict[str, float] = None
    
    # Paths
    dataset_path: str = "data/vqav2/train_index.json"
    checkpoint_dir: str = "checkpoints/subquestion_grpo"
    log_dir: str = "logs/subquestion_grpo"
    
    # Data parameters
    min_question_length: int = 5  # Minimum words for complex questions
    default_context: str = "car, person, building, street, vehicle, outdoor"  # Placeholder context
    max_training_samples: int = 200  # Maximum samples to use for training
    
    def __post_init__(self):
        if self.reward_weights is None:
            self.reward_weights = {
                "diversity": 0.20,      # How different questions are from each other
                "relevance": 0.25,      # How relevant to original question
                "answerability": 0.25,  # Can sub-Qs be answered independently?
                "completeness": 0.20,   # Do sub-Qs cover the original question?
                "clarity": 0.10,        # Are sub-Qs clear and well-formed?
            }


class JudgeLLM:
    """Judge LLM for scoring sub-questions."""
    
    def __init__(self, model_name: str):
        """Initialize judge model.
        
        Args:
            model_name: HuggingFace model name for judge
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Judge LLM: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def score_diversity(self, sub_questions: List[str]) -> float:
        """Score how diverse/different the sub-questions are from each other.
        
        Args:
            sub_questions: List of generated sub-questions
            
        Returns:
            Diversity score from 1-10
        """
        prompt = f"""Rate the diversity of these sub-questions on a scale of 1-10, where:
- 1-3: Very similar or repetitive questions
- 4-6: Some variation but overlapping concepts
- 7-9: Diverse questions covering different aspects
- 10: Highly diverse, each question explores a unique angle

Sub-questions:
{chr(10).join(f"{i+1}. {q}" for i, q in enumerate(sub_questions))}

Provide only a single number from 1 to 10 as your rating."""

        response = self._generate(prompt, max_new_tokens=10)
        return self._extract_score(response)
    
    def score_relevance(self, original_question: str, sub_questions: List[str], 
                       context: str) -> float:
        """Score how relevant sub-questions are to the original question.
        
        Args:
            original_question: The original complex question
            sub_questions: List of generated sub-questions
            context: Image context from CLIP
            
        Returns:
            Relevance score from 1-10
        """
        prompt = f"""Rate the relevance of these sub-questions to the original question on a scale of 1-10, where:
- 1-3: Sub-questions are off-topic or unrelated
- 4-6: Partially relevant but miss key aspects
- 7-9: Highly relevant, addressing the core question
- 10: Perfect relevance, directly supporting the answer

Image Context: {context}

Original Question: {original_question}

Sub-questions:
{chr(10).join(f"{i+1}. {q}" for i, q in enumerate(sub_questions))}

Provide only a single number from 1 to 10 as your rating."""

        response = self._generate(prompt, max_new_tokens=10)
        return self._extract_score(response)
    
    def score_answerability(self, sub_questions: List[str], context: str,
                           ground_truth: str = None) -> float:
        """Score whether sub-questions can be answered independently with tools.
        
        Args:
            sub_questions: List of generated sub-questions
            context: Image context from CLIP
            ground_truth: Optional ground truth answer as hint
            
        Returns:
            Answerability score from 1-10
        """
        gt_hint = f"\nGround Truth Answer (for reference): {ground_truth}" if ground_truth else ""
        
        prompt = f"""Rate how well these sub-questions can be answered independently using vision tools (object detection, OCR) on a scale of 1-10, where:
- 1-3: Questions require complex reasoning or external knowledge
- 4-6: Questions are answerable but require multiple steps
- 7-9: Questions can be directly answered with available tools
- 10: All questions are atomic and perfectly answerable

Image Context: {context}{gt_hint}

Sub-questions:
{chr(10).join(f"{i+1}. {q}" for i, q in enumerate(sub_questions))}

Available tools: object detection (grounding_dino), text recognition (OCR)

Provide only a single number from 1 to 10 as your rating."""

        response = self._generate(prompt, max_new_tokens=10)
        return self._extract_score(response)
    
    def score_completeness(self, original_question: str, sub_questions: List[str],
                          ground_truth: str = None) -> float:
        """Score whether answering the sub-questions would answer the original.
        
        Args:
            original_question: The original complex question
            sub_questions: List of generated sub-questions
            ground_truth: Optional ground truth answer for validation
            
        Returns:
            Completeness score from 1-10
        """
        gt_hint = f"\nGround Truth Answer (for validation): {ground_truth}" if ground_truth else ""
        
        prompt = f"""Rate how completely these sub-questions cover what's needed to answer the original question on a scale of 1-10, where:
- 1-3: Major gaps, missing critical information
- 4-6: Partial coverage, some aspects missing
- 7-9: Good coverage, most information present
- 10: Complete coverage, answering all sub-questions gives full answer

Original Question: {original_question}{gt_hint}

Sub-questions:
{chr(10).join(f"{i+1}. {q}" for i, q in enumerate(sub_questions))}

Provide only a single number from 1 to 10 as your rating."""

        response = self._generate(prompt, max_new_tokens=10)
        return self._extract_score(response)
    
    def score_clarity(self, sub_questions: List[str]) -> float:
        """Score the clarity and well-formedness of sub-questions.
        
        Args:
            sub_questions: List of generated sub-questions
            
        Returns:
            Clarity score from 1-10
        """
        prompt = f"""Rate the clarity and well-formedness of these sub-questions on a scale of 1-10, where:
- 1-3: Unclear, grammatically incorrect, or confusing
- 4-6: Understandable but awkwardly phrased
- 7-9: Clear and well-formed questions
- 10: Perfectly clear, concise, and well-structured

Sub-questions:
{chr(10).join(f"{i+1}. {q}" for i, q in enumerate(sub_questions))}

Provide only a single number from 1 to 10 as your rating."""

        response = self._generate(prompt, max_new_tokens=10)
        return self._extract_score(response)
    
    def compute_reward(self, original_question: str, sub_questions: List[str],
                      context: str, ground_truth: str = None,
                      weights: Dict[str, float] = None) -> Dict[str, float]:
        """Compute overall reward for generated sub-questions.
        
        Args:
            original_question: The original complex question
            sub_questions: List of generated sub-questions
            context: Image context from CLIP
            ground_truth: Optional ground truth answer
            weights: Weights for different criteria
            
        Returns:
            Dictionary with individual scores and total reward
        """
        if weights is None:
            weights = TrainingConfig().reward_weights
        
        # Compute individual scores
        scores = {
            "diversity": self.score_diversity(sub_questions),
            "relevance": self.score_relevance(original_question, sub_questions, context),
            "answerability": self.score_answerability(sub_questions, context, ground_truth),
            "completeness": self.score_completeness(original_question, sub_questions, ground_truth),
            "clarity": self.score_clarity(sub_questions),
        }
        
        # Compute weighted total (normalized to 0-1 scale)
        total_reward = sum(scores[k] * weights[k] for k in scores.keys()) / 10.0
        scores["total"] = total_reward
        
        return scores
    
    def _generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        """Generate response from judge model."""
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.3,  # Lower temperature for more consistent judging
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        input_length = inputs['input_ids'].shape[1]
        generated = outputs[0][input_length:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
    
    def _extract_score(self, response: str) -> float:
        """Extract numeric score from judge response."""
        import re
        # Look for numbers 1-10
        numbers = re.findall(r'\b([1-9]|10)\b', response)
        if numbers:
            return float(numbers[0])
        # Default to middle score if parsing fails
        return 5.0


class SubQuestionGRPOTrainer:
    """GRPO trainer for sub-question generation."""
    
    def __init__(self, config: TrainingConfig):
        """Initialize GRPO trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load policy model (model to train)
        print(f"Loading policy model: {config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
        self.policy_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Create reference model (frozen copy for KL divergence)
        print("Creating reference model...")
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        # Initialize judge LLM
        self.judge = JudgeLLM(config.judge_model_name)
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=config.learning_rate
        )
        
        # Create directories
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        
        # Training metrics
        self.metrics = []
    
    def generate_sub_questions(self, question: str, context: str, 
                              num_samples: int = 1) -> List[List[str]]:
        """Generate multiple samples of sub-questions using policy model.
        
        Args:
            question: Original question
            context: Image context from CLIP
            num_samples: Number of samples to generate
            
        Returns:
            List of sub-question lists
        """
        prompt = config.SUB_QUESTION_GENERATION_PROMPT.format(
            context=context,
            question=question
        )
        
        # Prepare input
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        all_sub_questions = []
        
        for _ in range(num_samples):
            with torch.no_grad():
                outputs = self.policy_model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    output_scores=True
                )
            
            input_length = inputs['input_ids'].shape[1]
            generated = outputs.sequences[0][input_length:]
            response = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
            
            # Parse sub-questions
            sub_questions = self._parse_sub_questions(response)
            all_sub_questions.append(sub_questions)
        
        return all_sub_questions
    
    def _parse_sub_questions(self, response: str) -> List[str]:
        """Parse sub-questions from model output."""
        import re
        
        sub_questions = [
            line.strip() 
            for line in response.split('\n') 
            if line.strip() and not line.strip().startswith('#')
        ]
        
        # Remove numbering
        sub_questions = [re.sub(r'^\d+\.?\s*', '', q) for q in sub_questions]
        sub_questions = [re.sub(r'^[-•\*]\s*', '', q) for q in sub_questions]
        
        # Filter valid questions
        sub_questions = [q for q in sub_questions if len(q) > 5]
        
        return sub_questions if sub_questions else ["What is in the image?"]
    
    def compute_grpo_loss(self, question: str, context: str, ground_truth: str = None):
        """Compute GRPO loss for one training example.
        
        Args:
            question: Original question
            context: Image context
            ground_truth: Optional ground truth answer
            
        Returns:
            Loss tensor and metrics dict
        """
        # Generate group_size samples
        samples = self.generate_sub_questions(
            question, context, num_samples=self.config.group_size
        )
        
        # Compute rewards for each sample
        rewards = []
        reward_details = []
        for sub_qs in samples:
            reward_dict = self.judge.compute_reward(
                question, sub_qs, context, ground_truth, self.config.reward_weights
            )
            rewards.append(reward_dict["total"])
            reward_details.append(reward_dict)
        
        rewards = torch.tensor(rewards, device=self.device)
        
        # Normalize rewards (advantage estimation)
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # Compute policy and reference log probabilities for each sample
        policy_logprobs = []
        ref_logprobs = []
        
        for sub_qs in samples:
            # Reconstruct prompt and response
            prompt = config.SUB_QUESTION_GENERATION_PROMPT.format(
                context=context, question=question
            )
            response = "\n".join(sub_qs)
            
            # Get log probabilities
            policy_lp = self._compute_logprobs(self.policy_model, prompt, response)
            ref_lp = self._compute_logprobs(self.ref_model, prompt, response)
            
            policy_logprobs.append(policy_lp)
            ref_logprobs.append(ref_lp)
        
        policy_logprobs = torch.stack(policy_logprobs)
        ref_logprobs = torch.stack(ref_logprobs)
        
        # Compute KL divergence
        kl_div = (policy_logprobs - ref_logprobs).mean()
        
        # GRPO loss: -E[advantage * log_prob] + KL_penalty
        policy_loss = -(advantages * policy_logprobs).mean()
        kl_penalty = self.config.kl_coef * kl_div
        
        total_loss = policy_loss + kl_penalty
        
        metrics = {
            "loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "kl_div": kl_div.item(),
            "mean_reward": rewards.mean().item(),
            "max_reward": rewards.max().item(),
            "min_reward": rewards.min().item(),
            "reward_details": reward_details[rewards.argmax().item()]  # Best sample's details
        }
        
        return total_loss, metrics
    
    def _compute_logprobs(self, model, prompt: str, response: str) -> torch.Tensor:
        """Compute log probabilities of response given prompt."""
        # Prepare full text
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        full_text = text + response
        
        # Tokenize
        inputs = self.tokenizer(full_text, return_tensors="pt")
        prompt_inputs = self.tokenizer(text, return_tensors="pt")
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        prompt_length = prompt_inputs['input_ids'].shape[1]
        
        # Get model outputs with appropriate context
        if model == self.ref_model:
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
        else:
            outputs = model(**inputs)
            logits = outputs.logits
        
        # Compute log probabilities for response tokens only
        response_logits = logits[0, prompt_length-1:-1, :]  # Shift for next token prediction
        response_tokens = inputs['input_ids'][0, prompt_length:]
        
        # Get log probabilities
        log_probs = F.log_softmax(response_logits, dim=-1)
        token_log_probs = log_probs.gather(1, response_tokens.unsqueeze(1)).squeeze(1)
        
        # Return mean log probability
        return token_log_probs.mean()
    
    def train_epoch(self, dataset: List[Dict], epoch: int):
        """Train for one epoch.
        
        Args:
            dataset: List of training examples
            epoch: Current epoch number
        """
        self.policy_model.train()
        epoch_metrics = []
        
        pbar = tqdm(dataset, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
        
        for i, example in enumerate(pbar):
            try:
                # Compute loss
                loss, metrics = self.compute_grpo_loss(
                    question=example["question"],
                    context=example.get("context", ""),
                    ground_truth=example.get("answer", None)
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
                self.optimizer.step()
                
                # Log metrics
                epoch_metrics.append(metrics)
                
                # Update progress bar
                pbar.set_postfix({
                    "loss": f"{metrics['loss']:.4f}",
                    "reward": f"{metrics['mean_reward']:.3f}"
                })
                
                # Save checkpoint periodically
                if (i + 1) % 100 == 0:
                    self.save_checkpoint(epoch, i)
                
            except Exception as e:
                print(f"Error processing example {i}: {e}")
                continue
        
        # Log epoch metrics
        self.log_epoch_metrics(epoch, epoch_metrics)
        
        return epoch_metrics
    
    def train(self, dataset: List[Dict]):
        """Train the model.
        
        Args:
            dataset: List of training examples with questions, contexts, and answers
        """
        print(f"Starting GRPO training for {self.config.num_epochs} epochs")
        print(f"Dataset size: {len(dataset)}")
        print(f"Batch size: {self.config.batch_size}, Group size: {self.config.group_size}")
        
        for epoch in range(self.config.num_epochs):
            epoch_metrics = self.train_epoch(dataset, epoch)
            
            # Save checkpoint after each epoch
            self.save_checkpoint(epoch, len(dataset))
            
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Avg Loss: {np.mean([m['loss'] for m in epoch_metrics]):.4f}")
            print(f"  Avg Reward: {np.mean([m['mean_reward'] for m in epoch_metrics]):.3f}")
            print(f"  Max Reward: {np.max([m['max_reward'] for m in epoch_metrics]):.3f}")
        
        print("\nTraining completed!")
        self.save_final_model()
    
    def save_checkpoint(self, epoch: int, step: int):
        """Save training checkpoint."""
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir, 
            f"checkpoint_epoch{epoch}_step{step}.pt"
        )
        torch.save({
            'epoch': epoch,
            'step': step,
            'model_state_dict': self.policy_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
        }, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
    
    def save_final_model(self):
        """Save final trained model."""
        model_path = os.path.join(self.config.checkpoint_dir, "final_model")
        self.policy_model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)
        print(f"Saved final model to: {model_path}")
    
    def log_epoch_metrics(self, epoch: int, metrics: List[Dict]):
        """Log metrics for the epoch."""
        log_path = os.path.join(self.config.log_dir, f"epoch_{epoch}_metrics.json")
        with open(log_path, 'w') as f:
            json.dump(metrics, f, indent=2)


def load_training_data(dataset_path: str, max_samples: int = None, 
                      min_question_length: int = 5,
                      default_context: str = "car, person, building, street, vehicle, outdoor") -> List[Dict]:
    """Load training data from VQAv2 dataset.
    
    Args:
        dataset_path: Path to dataset index JSON
        max_samples: Maximum number of samples to load
        min_question_length: Minimum words in question to consider it complex
        default_context: Default CLIP context when not available
        
    Returns:
        List of training examples
    """
    print(f"Loading training data from: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print(f"Warning: Dataset not found at {dataset_path}")
        print("Generating dummy examples for demonstration...")
        return generate_dummy_data(100)
    
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    # Extract complex questions (longer questions more likely to need decomposition)
    training_examples = []
    for item in data:
        question = item.get("question", "")
        # Select questions with more than min_question_length words as potentially complex
        if len(question.split()) > min_question_length:
            training_examples.append({
                "question": question,
                "answer": item.get("answer", ""),
                "context": item.get("context", default_context),
                "image_id": item.get("image_id", "")
            })
        
        if max_samples and len(training_examples) >= max_samples:
            break
    
    print(f"Loaded {len(training_examples)} training examples")
    return training_examples


def generate_dummy_data(num_samples: int = 100) -> List[Dict]:
    """Generate dummy training data for testing.
    
    Args:
        num_samples: Number of dummy examples
        
    Returns:
        List of dummy training examples
    """
    questions = [
        "How many people are standing near the red car?",
        "What color is the sign behind the person on the left?",
        "Are there more cars or people in the image?",
        "What text is written on the building in the background?",
        "How many objects are visible on the table?",
        "Is the person wearing a hat and glasses?",
        "What is the position of the dog relative to the car?",
        "How many windows does the building have?",
        "What is the person holding in their left hand?",
        "Are there any animals near the tree?",
    ]
    
    contexts = [
        "car, person, street, vehicle, outdoor, red, blue",
        "sign, text, person, building, urban",
        "car, person, vehicle, people, crowd",
        "building, text, sign, architecture, outdoor",
        "table, object, item, indoor, furniture",
    ]
    
    data = []
    for i in range(num_samples):
        data.append({
            "question": questions[i % len(questions)],
            "answer": "dummy answer",
            "context": contexts[i % len(contexts)],
            "image_id": f"dummy_{i}"
        })
    
    return data


def main():
    """Main training function."""
    # Initialize configuration
    config = TrainingConfig()
    
    print("=" * 60)
    print("GRPO Training for Sub-Question Generation")
    print("=" * 60)
    print(f"Policy Model: {config.model_name}")
    print(f"Judge Model: {config.judge_model_name}")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Group Size: {config.group_size}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Reward Weights: {config.reward_weights}")
    print("=" * 60)
    
    # Load training data
    dataset = load_training_data(
        config.dataset_path, 
        max_samples=config.max_training_samples,
        min_question_length=config.min_question_length,
        default_context=config.default_context
    )
    
    # Initialize trainer
    trainer = SubQuestionGRPOTrainer(config)
    
    # Train
    trainer.train(dataset)
    
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
