"""Example script demonstrating HTML report generation for VQA inference."""

from inference import VQAInference
from report_generator import ReportingVQAInference
from PIL import Image
import numpy as np

def create_sample_image(width=800, height=600):
    """Create a sample image for demonstration."""
    # Create a simple test image
    img_array = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Add a colored rectangle (car)
    img_array[200:450, 120:380] = [200, 50, 50]  # Red rectangle
    
    # Add text area (sign)
    img_array[100:200, 600:750] = [255, 50, 50]  # Red square for sign
    
    # Add some colored circles (people)
    for cx, cy in [(500, 350), (550, 350)]:
        for y in range(cy-30, cy+30):
            for x in range(cx-20, cx+20):
                if 0 <= y < height and 0 <= x < width:
                    if (x-cx)**2 + (y-cy)**2 < 400:
                        img_array[y, x] = [100, 100, 200]  # Blue circles
    
    img = Image.fromarray(img_array)
    return img


def main():
    """Run examples with HTML report generation."""
    
    print("=" * 80)
    print("VQA Inference with HTML Report Generation")
    print("=" * 80)
    
    # Create sample image
    print("\nCreating sample image...")
    sample_image = create_sample_image()
    sample_image_path = "sample_scene.png"
    sample_image.save(sample_image_path)
    print(f"Sample image saved to: {sample_image_path}")
    
    # Initialize VQA inference engine
    print("\nInitializing VQA Inference Engine...")
    base_vqa = VQAInference(max_depth=3)
    
    # Wrap with reporting capability
    vqa_with_report = ReportingVQAInference(base_vqa)
    
    # Example 1: Simple atomic question
    print("\n" + "=" * 80)
    print("Example 1: Simple Atomic Question")
    print("=" * 80)
    
    result1 = vqa_with_report.inference_vqav2_with_report(
        image_path=sample_image_path,
        question="What color is the car?",
        report_path="report_example1_atomic.html"
    )
    
    # Example 2: Question requiring multiple tools
    print("\n" + "=" * 80)
    print("Example 2: Multiple Tool Calls")
    print("=" * 80)
    
    result2 = vqa_with_report.inference_vqav2_with_report(
        image_path=sample_image_path,
        question="What objects and text are visible in the image?",
        report_path="report_example2_multiple_tools.html"
    )
    
    # Example 3: Complex question with recursion
    print("\n" + "=" * 80)
    print("Example 3: Complex Question with Recursion")
    print("=" * 80)
    
    result3 = vqa_with_report.inference_vqav2_with_report(
        image_path=sample_image_path,
        question="How many people are standing near the red car?",
        report_path="report_example3_recursive.html"
    )
    
    # Summary
    print("\n" + "=" * 80)
    print("All Examples Complete!")
    print("=" * 80)
    print("\nGenerated HTML Reports:")
    print(f"  1. {result1['report_path']} - Simple atomic question")
    print(f"  2. {result2['report_path']} - Multiple tool calls")
    print(f"  3. {result3['report_path']} - Complex recursive question")
    print("\nOpen these HTML files in your browser to view the detailed reports.")
    print("Each report includes:")
    print("  - Input image visualization")
    print("  - Question and final answer")
    print("  - Step-by-step inference process")
    print("  - CLIP visual context")
    print("  - Atomicity checks")
    print("  - Tool calls and results")
    print("  - Sub-question generation and answers")
    print("  - LLM reasoning steps")
    print("=" * 80)


if __name__ == "__main__":
    main()
