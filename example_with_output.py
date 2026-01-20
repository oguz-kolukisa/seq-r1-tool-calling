"""
Example demonstrating VQAv2 inference with complete input/output flow.
This shows what happens when you process an image with questions.
"""


def demonstrate_inference_flow():
    """Demonstrate the complete inference flow with example outputs."""
    
    print("="*80)
    print("VQAv2 Inference System - Complete Example")
    print("="*80)
    
    # Example image description
    print("\n1. INPUT IMAGE")
    print("-" * 80)
    print("Image: sample_scene.jpg")
    print("Image size: 800x600 pixels")
    print("Image contains:")
    print("  - Red car in the left side")
    print("  - Two people standing near the car")
    print("  - A stop sign with text 'STOP'")
    print("  - Blue building in the background")
    
    # Example 1: Atomic question with single tool call
    print("\n" + "="*80)
    print("EXAMPLE 1: Atomic Question - Single Tool Call")
    print("="*80)
    
    question1 = "What color is the car?"
    print(f"\n2. INPUT QUESTION: '{question1}'")
    print("-" * 80)
    
    print("\n3. PROCESSING FLOW:")
    print("-" * 80)
    print("[Depth 0] Answering: What color is the car?")
    print("  → Extracting visual context with CLIP...")
    print("  → Context: car, red, vehicle, outdoor, road, person, sign, building, scene, street")
    print("  → Checking atomicity...")
    print("  → Is atomic: True")
    print("  → Generating tool calls...")
    print("  → Tool calls: ['grounding_dino(query=\"car\")']")
    print("  → Executing tool: grounding_dino(query=\"car\")")
    print("  → Tool result: Detected 1 instances of 'car': [{'box': [120, 200, 380, 450], 'score': 0.95, 'label': 'car'}]")
    print("  → Reasoning from 1 tool result(s)...")
    print("  → Final answer: The car is red.")
    
    print("\n4. OUTPUT:")
    print("-" * 80)
    print(f"Question: {question1}")
    print(f"Answer: The car is red.")
    print(f"Tool calls executed: 1")
    print(f"Recursion depth: 0")
    
    # Example 2: Atomic question with multiple tool calls
    print("\n" + "="*80)
    print("EXAMPLE 2: Atomic Question - Multiple Tool Calls")
    print("="*80)
    
    question2 = "What objects and text are visible?"
    print(f"\n2. INPUT QUESTION: '{question2}'")
    print("-" * 80)
    
    print("\n3. PROCESSING FLOW:")
    print("-" * 80)
    print("[Depth 0] Answering: What objects and text are visible?")
    print("  → Extracting visual context with CLIP...")
    print("  → Context: car, person, sign, text, stop, building, outdoor, vehicle, people, road")
    print("  → Checking atomicity...")
    print("  → Is atomic: True")
    print("  → Generating tool calls...")
    print("  → Tool calls: ['grounding_dino(query=\"object\")', 'ocr()']")
    print("  → Executing tool: grounding_dino(query=\"object\")")
    print("  → Tool result: Detected 4 instances of 'object': [{'box': [120, 200, 380, 450], 'score': 0.92, 'label': 'car'}, {'box': [450, 250, 520, 480], 'score': 0.89, 'label': 'person'}, {'box': [470, 255, 535, 485], 'score': 0.87, 'label': 'person'}, {'box': [600, 150, 680, 280], 'score': 0.85, 'label': 'sign'}]")
    print("  → Executing tool: ocr()")
    print("  → Tool result: OCR detected 1 text regions. Full text: 'STOP'")
    print("  → Reasoning from 2 tool result(s)...")
    print("  → Final answer: The image contains a red car, two people, and a stop sign. The visible text on the sign reads 'STOP'.")
    
    print("\n4. OUTPUT:")
    print("-" * 80)
    print(f"Question: {question2}")
    print(f"Answer: The image contains a red car, two people, and a stop sign. The visible text on the sign reads 'STOP'.")
    print(f"Tool calls executed: 2")
    print(f"Recursion depth: 0")
    
    # Example 3: Complex question with recursion
    print("\n" + "="*80)
    print("EXAMPLE 3: Complex Question - Recursive Decomposition")
    print("="*80)
    
    question3 = "How many people are near the car?"
    print(f"\n2. INPUT QUESTION: '{question3}'")
    print("-" * 80)
    
    print("\n3. PROCESSING FLOW:")
    print("-" * 80)
    print("[Depth 0] Answering: How many people are near the car?")
    print("  → Extracting visual context with CLIP...")
    print("  → Context: people, person, car, near, count, vehicle, standing, outdoor, scene")
    print("  → Checking atomicity...")
    print("  → Is atomic: False (complex question with multiple parts)")
    print("  → Generating sub-questions...")
    print("  → Sub-questions: ['Where is the car?', 'Where are the people?', 'How many people are there?']")
    print()
    print("  [Depth 1] Answering: Where is the car?")
    print("    → CLIP context: car, vehicle, location, position")
    print("    → Is atomic: True")
    print("    → Tool calls: ['grounding_dino(query=\"car\")']")
    print("    → Tool result: Detected 1 instances of 'car': [{'box': [120, 200, 380, 450], 'score': 0.95}]")
    print("    → Final answer: The car is located on the left side of the image.")
    print()
    print("  [Depth 1] Answering: Where are the people?")
    print("    → CLIP context: people, person, location, standing")
    print("    → Is atomic: True")
    print("    → Tool calls: ['grounding_dino(query=\"person\")']")
    print("    → Tool result: Detected 2 instances of 'person': [{'box': [450, 250, 520, 480], 'score': 0.89}, {'box': [470, 255, 535, 485], 'score': 0.87}]")
    print("    → Final answer: There are people standing near the car on the right side.")
    print()
    print("  [Depth 1] Answering: How many people are there?")
    print("    → CLIP context: people, count, number, person")
    print("    → Is atomic: True")
    print("    → Tool calls: ['grounding_dino(query=\"person\")']")
    print("    → Tool result: Detected 2 instances of 'person'")
    print("    → Final answer: There are 2 people.")
    print()
    print("  [Depth 0] Aggregating results...")
    print("  → Final answer: There are 2 people standing near the car.")
    
    print("\n4. OUTPUT:")
    print("-" * 80)
    print(f"Question: {question3}")
    print(f"Answer: There are 2 people standing near the car.")
    print(f"Sub-questions processed: 3")
    print(f"Total tool calls executed: 3")
    print(f"Max recursion depth: 1")
    
    # Summary
    print("\n" + "="*80)
    print("SYSTEM CAPABILITIES DEMONSTRATED")
    print("="*80)
    print("✓ CLIP visual context extraction (top-10 concepts)")
    print("✓ LLM-based atomicity checking")
    print("✓ Single tool call generation")
    print("✓ Multiple tool calls generation (NEW)")
    print("✓ Grounding DINO for object detection")
    print("✓ OCR for text recognition")
    print("✓ Recursive question decomposition")
    print("✓ LLM reasoning for final answers")
    print("✓ Result aggregation from multiple sources")
    
    print("\n" + "="*80)
    print("KEY IMPROVEMENTS")
    print("="*80)
    print("1. Multiple Tool Calls: System can now call multiple tools for complex atomic questions")
    print("2. Better Coverage: Questions requiring both detection and OCR are handled in one step")
    print("3. LLM Validation: All outputs go through LLM for natural language generation")
    print("="*80)


def usage_example():
    """Show how to use the system in code."""
    print("\n" + "="*80)
    print("CODE USAGE EXAMPLE")
    print("="*80)
    print("""
# Import the system
from inference import VQAInference
from PIL import Image

# Initialize
vqa = VQAInference(max_depth=3)

# Load your image
image = Image.open("your_image.jpg")

# Ask a question
result = vqa.inference_vqav2(
    image_path="your_image.jpg",
    question="What objects and text are visible?"
)

# The system will:
# 1. Extract CLIP visual context
# 2. Check if question is atomic
# 3. Generate multiple tool calls if needed
# 4. Execute all tools (Grounding DINO, OCR, etc.)
# 5. Use LLM to reason about results
# 6. Return natural language answer

print(f"Answer: {result['answer']}")
    """)


if __name__ == "__main__":
    demonstrate_inference_flow()
    usage_example()
    
    print("\n" + "="*80)
    print("To run the actual system with real models:")
    print("  python run_inference.py --image image.jpg --question 'Your question?'")
    print("="*80)
