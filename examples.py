"""Usage examples for VQAv2 inference system."""

from inference import VQAInference


def example_basic():
    """Basic single inference example."""
    print("Example: Basic Inference")
    print("-" * 40)
    
    vqa = VQAInference(max_depth=3)
    result = vqa.inference_vqav2(
        image_path="path/to/image.jpg",
        question="What is in this image?"
    )
    print(f"Answer: {result['answer']}")


def example_batch():
    """Batch inference example."""
    print("\nExample: Batch Processing")
    print("-" * 40)
    
    vqa = VQAInference(max_depth=3)
    data = [
        ("image1.jpg", "How many people?"),
        ("image2.jpg", "What color is the car?"),
    ]
    results = vqa.batch_inference(data)
    
    for r in results:
        print(f"Q: {r['question']}")
        print(f"A: {r['answer']}\n")


def example_vqa2_dataset():
    """VQAv2 dataset example."""
    print("\nExample: VQAv2 Dataset")
    print("-" * 40)
    
    import json
    
    # Load dataset
    with open('data/vqav2/val_index.json') as f:
        data = json.load(f)
    
    vqa = VQAInference(max_depth=3)
    
    # Process first 3 samples
    for entry in data[:3]:
        result = vqa.inference_vqav2(
            entry['image_path'],
            entry['question']
        )
        print(f"Q: {entry['question']}")
        print(f"A: {result['answer']}")
        if 'answers' in entry:
            print(f"GT: {entry['answers'][:3]}")
        print()


if __name__ == "__main__":
    print("VQAv2 Inference Examples")
    print("=" * 40)
    print("\nNote: Update image paths to run examples")
    print("=" * 40)
    
    # Uncomment to run:
    # example_basic()
    # example_batch()
    # example_vqa2_dataset()

