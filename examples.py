"""Example script showing how to use VQA inference system."""

from inference import VQAInference
from PIL import Image
import json


def example_single_inference():
    """Example of single question-image inference."""
    print("="*80)
    print("Example 1: Single Inference")
    print("="*80)
    
    # Initialize VQA engine with default models
    vqa_engine = VQAInference(max_depth=3)
    
    # Example inference (you need to provide actual image path)
    image_path = "path/to/your/image.jpg"
    question = "How many people are in the image?"
    
    # Run inference
    result = vqa_engine.inference_vqav2(image_path, question)
    
    print("\nResult:")
    print(json.dumps(result, indent=2))


def example_batch_inference():
    """Example of batch inference on multiple questions."""
    print("\n" + "="*80)
    print("Example 2: Batch Inference")
    print("="*80)
    
    # Initialize VQA engine
    vqa_engine = VQAInference(max_depth=3)
    
    # Prepare batch data (list of (image_path, question) tuples)
    data = [
        ("path/to/image1.jpg", "What color is the car?"),
        ("path/to/image2.jpg", "How many people are sitting at the table?"),
        ("path/to/image3.jpg", "What text is visible in the image?"),
    ]
    
    # Run batch inference
    results = vqa_engine.batch_inference(data)
    
    print("\nBatch Results:")
    print(json.dumps(results, indent=2))


def example_custom_models():
    """Example using custom model configurations."""
    print("\n" + "="*80)
    print("Example 3: Custom Models")
    print("="*80)
    
    # Initialize with custom models
    vqa_engine = VQAInference(
        llm_model_name="microsoft/phi-2",  # Smaller LLM for faster inference
        clip_model_name="openai/clip-vit-base-patch32",
        max_depth=2  # Limit recursion depth
    )
    
    image_path = "path/to/image.jpg"
    question = "What is happening in this scene?"
    
    result = vqa_engine.inference_vqav2(image_path, question)
    
    print("\nResult with custom models:")
    print(json.dumps(result, indent=2))


def example_direct_answer_function():
    """Example of directly using the recursive answer function."""
    print("\n" + "="*80)
    print("Example 4: Direct Answer Function Usage")
    print("="*80)
    
    # Initialize VQA engine
    vqa_engine = VQAInference(max_depth=3)
    
    # Load image
    image_path = "path/to/image.jpg"
    image = Image.open(image_path).convert('RGB')
    
    # Complex question that should trigger sub-question generation
    question = "How many red objects are there and what is their location?"
    
    # Call answer function directly
    answer = vqa_engine.answer(image, question, depth=0)
    
    print(f"\nQuestion: {question}")
    print(f"Answer: {answer}")


if __name__ == "__main__":
    print("VQAv2 Inference Examples")
    print("="*80)
    print("\nThese examples demonstrate the VQA inference system.")
    print("Note: You need to provide actual image paths to run these examples.")
    print("="*80)
    
    # Uncomment the example you want to run:
    # example_single_inference()
    # example_batch_inference()
    # example_custom_models()
    # example_direct_answer_function()
    
    print("\nTo run examples, uncomment the desired function call in the script.")
