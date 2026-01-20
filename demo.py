"""Demo showing the recursive inference flow."""


def show_flow():
    """Show how the system works."""
    print("="*60)
    print("VQAv2 Recursive Inference Demo")
    print("="*60)
    
    print("\n1. ATOMIC QUESTION")
    print("-" * 40)
    print("Q: What color is the car?")
    print()
    print("Flow:")
    print("  [Depth 0] → CLIP context → ATOMIC")
    print("          → grounding_dino(query='car')")
    print("          → 'Red car detected'")
    print()
    print("A: The car is red")
    
    print("\n2. COMPLEX QUESTION (Recursion)")
    print("-" * 40)
    print("Q: How many people wearing red shirts?")
    print()
    print("Flow:")
    print("  [Depth 0] → NOT_ATOMIC")
    print("          → Sub-questions:")
    print("              1. 'Where are people?'")
    print("              2. 'Who wears red?'")
    print()
    print("  [Depth 1] → 'Where are people?'")
    print("          → ATOMIC → grounding_dino(query='person')")
    print("          → '5 people detected'")
    print()
    print("  [Depth 1] → 'Who wears red?'")
    print("          → ATOMIC → grounding_dino(query='red shirt')")
    print("          → '3 with red shirts'")
    print()
    print("  [Depth 0] → Aggregate")
    print("          → '3 people wearing red shirts'")
    print()
    print("A: 3 people wearing red shirts")
    
    print("\n" + "="*60)
    print("Key Concepts:")
    print("  • CLIP extracts visual context (top-10 concepts)")
    print("  • LLM determines if question is atomic")
    print("  • Atomic → tool call, Not atomic → decompose")
    print("  • Recursive processing with depth limit (default: 3)")
    print("  • Results aggregated by LLM")
    print("="*60)


if __name__ == "__main__":
    show_flow()

