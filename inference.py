"""Main inference script for VQAv2 with recursive tool calling."""

from PIL import Image
from typing import Optional, List, Tuple
import config
from clip_extractor import CLIPContextExtractor
from llm_wrapper import LLMWrapper
from tools import ToolExecutor


class VQAInference:
    """VQA inference engine with recursive tool calling."""
    
    def __init__(
        self,
        llm_model_name: Optional[str] = None,
        clip_model_name: Optional[str] = None,
        max_depth: int = config.DEFAULT_DEPTH
    ):
        """Initialize VQA inference engine.
        
        Args:
            llm_model_name: Optional LLM model name (uses config default if None)
            clip_model_name: Optional CLIP model name (uses config default if None)
            max_depth: Maximum recursion depth for answer function
        """
        print("Initializing VQA Inference Engine...")
        
        self.max_depth = max_depth
        
        # Initialize CLIP for context extraction
        print("Loading CLIP model...")
        clip_name = clip_model_name or config.CLIP_MODEL_NAME
        self.clip_extractor = CLIPContextExtractor(clip_name)
        
        # Initialize LLM
        print("Loading LLM...")
        llm_name = llm_model_name or config.LLM_MODEL_NAME
        self.llm = LLMWrapper(llm_name)
        
        # Initialize tool executor
        print("Initializing tools...")
        self.tool_executor = ToolExecutor()
        
        print("VQA Inference Engine initialized successfully!")
    
    def answer(
        self,
        image: Image.Image,
        question: str,
        depth: int = 0
    ) -> str:
        """Recursive answer function for VQA.
        
        This is the main recursive function that:
        1. Checks if question is atomic
        2. If atomic: generates and executes tool call
        3. If not atomic: generates sub-questions and recursively answers them
        4. Aggregates results from sub-questions
        
        Args:
            image: PIL Image object
            question: Question to answer
            depth: Current recursion depth
            
        Returns:
            Answer to the question
        """
        print(f"\n{'  ' * depth}[Depth {depth}] Answering: {question}")
        
        # Check depth limit
        if depth >= self.max_depth:
            print(f"{'  ' * depth}Max depth reached. Returning default answer.")
            return f"Maximum recursion depth ({self.max_depth}) reached. Unable to fully answer the question."
        
        # Step 1: Extract visual context using CLIP
        print(f"{'  ' * depth}Extracting visual context with CLIP...")
        context = self.clip_extractor.extract_context(image, question)
        print(f"{'  ' * depth}Context: {context}")
        
        # Step 2: Check atomicity
        print(f"{'  ' * depth}Checking atomicity...")
        is_atomic = self.llm.check_atomicity(question, context)
        print(f"{'  ' * depth}Is atomic: {is_atomic}")
        
        if is_atomic:
            # Step 3a: Generate tool calls (can be multiple)
            print(f"{'  ' * depth}Generating tool calls...")
            tool_calls = self.llm.generate_tool_call(question, context)
            print(f"{'  ' * depth}Tool calls: {tool_calls}")
            
            # Execute all tool calls
            tool_results = []
            for tool_call in tool_calls:
                print(f"{'  ' * depth}Executing tool: {tool_call}")
                tool_result = self.tool_executor.execute_tool_call(image, tool_call)
                print(f"{'  ' * depth}Tool result: {tool_result}")
                tool_results.append((tool_call, tool_result))
            
            # Reason about all tool results to generate final answer
            print(f"{'  ' * depth}Reasoning from {len(tool_results)} tool result(s)...")
            final_answer = self.llm.aggregate_results(
                question, 
                context, 
                tool_results
            )
            print(f"{'  ' * depth}Final answer: {final_answer}")
            
            return final_answer
        
        else:
            # Step 3b: Generate sub-questions
            print(f"{'  ' * depth}Generating sub-questions...")
            sub_questions = self.llm.generate_sub_questions(question, context)
            print(f"{'  ' * depth}Sub-questions: {sub_questions}")
            
            # Step 4: Recursively answer sub-questions
            sub_results = []
            for sub_q in sub_questions:
                sub_answer = self.answer(image, sub_q, depth + 1)
                sub_results.append((sub_q, sub_answer))
            
            # Step 5: Aggregate results
            print(f"{'  ' * depth}Aggregating results...")
            final_answer = self.llm.aggregate_results(question, context, sub_results)
            print(f"{'  ' * depth}Final answer: {final_answer}")
            
            return final_answer
    
    def inference_vqav2(
        self,
        image_path: str,
        question: str
    ) -> dict:
        """Run inference on a VQAv2 question-image pair.
        
        Args:
            image_path: Path to the image file
            question: Question about the image
            
        Returns:
            Dictionary containing question, answer, and metadata
        """
        print(f"\n{'='*80}")
        print(f"Processing VQAv2 Question: {question}")
        print(f"Image: {image_path}")
        print(f"{'='*80}")
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Get answer using recursive function
        answer = self.answer(image, question, depth=0)
        
        result = {
            "question": question,
            "image_path": image_path,
            "answer": answer,
            "max_depth": self.max_depth
        }
        
        print(f"\n{'='*80}")
        print(f"Final Result:")
        print(f"Question: {question}")
        print(f"Answer: {answer}")
        print(f"{'='*80}\n")
        
        return result
    
    def batch_inference(
        self,
        data: List[Tuple[str, str]]
    ) -> List[dict]:
        """Run inference on multiple VQAv2 question-image pairs.
        
        Args:
            data: List of (image_path, question) tuples
            
        Returns:
            List of result dictionaries
        """
        results = []
        
        for i, (image_path, question) in enumerate(data):
            print(f"\n{'#'*80}")
            print(f"Processing sample {i+1}/{len(data)}")
            print(f"{'#'*80}")
            
            try:
                result = self.inference_vqav2(image_path, question)
                results.append(result)
            except Exception as e:
                print(f"Error processing sample {i+1}: {e}")
                results.append({
                    "question": question,
                    "image_path": image_path,
                    "answer": f"Error: {e}",
                    "max_depth": self.max_depth
                })
        
        return results
