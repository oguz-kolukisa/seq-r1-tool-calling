"""Example usage script for VQAv2 inference."""

from inference import VQAInference
import argparse
import json
from pathlib import Path


def main():
    """Main function for running VQAv2 inference."""
    parser = argparse.ArgumentParser(description="VQAv2 Inference with Recursive Tool Calling")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="Question about the image"
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="HuggingFace LLM model name (optional, uses config default)"
    )
    parser.add_argument(
        "--clip-model",
        type=str,
        default=None,
        help="HuggingFace CLIP model name (optional, uses config default)"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Maximum recursion depth (default: 3)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save output JSON (optional)"
    )
    
    args = parser.parse_args()
    
    # Validate image path
    if not Path(args.image).exists():
        print(f"Error: Image file not found: {args.image}")
        return
    
    # Initialize inference engine
    vqa_engine = VQAInference(
        llm_model_name=args.llm_model,
        clip_model_name=args.clip_model,
        max_depth=args.max_depth
    )
    
    # Run inference
    result = vqa_engine.inference_vqav2(args.image, args.question)
    
    # Save output if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResult saved to: {args.output}")
    
    print("\nInference completed!")


if __name__ == "__main__":
    main()
