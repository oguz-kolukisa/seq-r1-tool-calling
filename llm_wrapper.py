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
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True
        )
        
        if not torch.cuda.is_available():
            self.model = self.model.to(self.device)
            
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate(self, prompt: str, max_length: int = 512, temperature: float = 0.7) -> str:
        """Generate text response from prompt.
        
        Args:
            prompt: Input prompt
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            
        Returns:
            Generated text response
        """
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the prompt from the generated text
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        return generated_text
    
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
        
        response = self.generate(prompt, max_length=256)
        
        # Parse response - looking for "ATOMIC" or "NOT_ATOMIC"
        response_upper = response.upper()
        if "ATOMIC" in response_upper and "NOT_ATOMIC" not in response_upper:
            return True
        return False
    
    def generate_tool_call(self, question: str, context: str) -> str:
        """Generate appropriate tool call for an atomic question.
        
        Args:
            question: The atomic question
            context: Visual context from CLIP
            
        Returns:
            Tool call string
        """
        prompt = config.TOOL_CALL_GENERATION_PROMPT.format(
            context=context,
            question=question
        )
        
        response = self.generate(prompt, max_length=256)
        
        # Extract tool call from response
        # Look for patterns like "grounding_dino(...)" or "ocr()"
        import re
        tool_pattern = r'(grounding_dino\([^)]*\)|ocr\(\))'
        match = re.search(tool_pattern, response)
        
        if match:
            return match.group(1)
        
        # If no tool call found, return the response as is
        return response.strip()
    
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
        
        response = self.generate(prompt, max_length=512)
        
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
        
        response = self.generate(prompt, max_length=512)
        
        return response.strip()
