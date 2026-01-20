"""LLM wrapper using HuggingFace Transformers."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional
import config


class LLMWrapper:
    """Wrapper for HuggingFace language models."""
    
    def __init__(self, model_name: str = config.LLM_MODEL_NAME):
        """Initialize LLM model and tokenizer.
        
        Args:
            model_name: HuggingFace model name
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        
        print(f"Loading LLM model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        if not torch.cuda.is_available():
            self.model = self.model.to(self.device)
            
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Check if model supports chat template
        self.supports_chat = hasattr(self.tokenizer, 'apply_chat_template')
    
    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.7) -> str:
        """Generate text response from prompt.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text response
        """
        # For chat models like Qwen, use chat template
        if self.supports_chat:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = self.tokenizer([text], return_tensors="pt")
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only the new tokens
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return generated_text.strip()
    
    def check_atomicity(self, question: str, context: str) -> bool:
        """Check if a question is atomic using the LLM.
        
        Args:
            question: The question to check
            context: Visual context from CLIP
            
        Returns:
            True if atomic, False otherwise
        """
        prompt = config.ATOMICITY_CHECK_PROMPT.format(
            context=context,
            question=question
        )
        
        response = self.generate(prompt, max_new_tokens=128)
        
        # Parse response - looking for "ATOMIC" or "NOT_ATOMIC"
        response_upper = response.upper()
        if "NOT_ATOMIC" in response_upper or "NOT ATOMIC" in response_upper:
            return False
        if "ATOMIC" in response_upper:
            return True
        # Default to atomic for simple questions
        return len(question.split()) <= 10
    
    def generate_tool_call(self, question: str, context: str) -> list:
        """Generate appropriate tool calls for an atomic question.
        
        Args:
            question: The atomic question
            context: Visual context from CLIP
            
        Returns:
            List of tool call strings (can be one or multiple)
        """
        prompt = config.TOOL_CALL_GENERATION_PROMPT.format(
            context=context,
            question=question
        )
        
        response = self.generate(prompt, max_new_tokens=256)
        
        # Extract all tool calls from response
        import re
        tool_pattern = r'(grounding_dino\([^)]*\)|ocr\(\))'
        matches = re.findall(tool_pattern, response, re.IGNORECASE)
        
        if matches:
            # Remove duplicates while preserving order
            seen = set()
            unique_calls = []
            for call in matches:
                if call.lower() not in seen:
                    seen.add(call.lower())
                    unique_calls.append(call)
            return unique_calls
        
        # Fallback: infer tool from question
        if any(word in question.lower() for word in ['text', 'read', 'write', 'written', 'say', 'sign']):
            return ['ocr()']
        else:
            # Extract noun/object from question for grounding
            words = question.lower().replace('?', '').split()
            # Look for key nouns
            for word in ['person', 'people', 'car', 'object', 'thing', 'animal', 'building']:
                if word in words:
                    return [f'grounding_dino(query="{word}")']
            return ['grounding_dino(query="object")']
    
    def generate_sub_questions(self, question: str, context: str) -> list:
        """Generate sub-questions for a complex question.
        
        Args:
            question: The complex question
            context: Visual context from CLIP
            
        Returns:
            List of sub-questions
        """
        prompt = config.SUB_QUESTION_GENERATION_PROMPT.format(
            context=context,
            question=question
        )
        
        response = self.generate(prompt, max_new_tokens=256)
        
        # Parse response - split by newlines and filter empty lines
        sub_questions = [
            line.strip() 
            for line in response.split('\n') 
            if line.strip() and not line.strip().startswith('#')
        ]
        
        # Remove numbering if present (e.g., "1.", "2.", etc.)
        import re
        sub_questions = [
            re.sub(r'^\d+\.?\s*', '', q) 
            for q in sub_questions
        ]
        
        # Remove any leading dashes or bullets
        sub_questions = [
            re.sub(r'^[-•\*]\s*', '', q)
            for q in sub_questions
        ]
        
        # Filter out empty or very short questions
        sub_questions = [q for q in sub_questions if len(q) > 5]
        
        # If no sub-questions generated, create default ones
        if len(sub_questions) == 0:
            sub_questions = [question]  # Use original question as fallback
        
        return sub_questions
    
    def aggregate_results(self, question: str, context: str, sub_results: list) -> str:
        """Aggregate results from sub-questions into final answer.
        
        Args:
            question: The original question
            context: Visual context from CLIP
            sub_results: List of (sub_question, answer) tuples
            
        Returns:
            Final aggregated answer
        """
        # Format sub-results
        sub_results_text = "\n".join([
            f"Q: {sub_q}\nA: {answer}"
            for sub_q, answer in sub_results
        ])
        
        prompt = config.AGGREGATE_RESULTS_PROMPT.format(
            question=question,
            context=context,
            sub_results=sub_results_text
        )
        
        response = self.generate(prompt, max_new_tokens=256)
        
        return response.strip()
