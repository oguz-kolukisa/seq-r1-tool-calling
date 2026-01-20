"""Simple test to verify the inference system structure."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        import config
        print("✓ config imported")
    except Exception as e:
        print(f"✗ config import failed: {e}")
        return False
    
    try:
        from clip_extractor import CLIPContextExtractor
        print("✓ clip_extractor imported")
    except Exception as e:
        print(f"✗ clip_extractor import failed: {e}")
        return False
    
    try:
        from llm_wrapper import LLMWrapper
        print("✓ llm_wrapper imported")
    except Exception as e:
        print(f"✗ llm_wrapper import failed: {e}")
        return False
    
    try:
        from tools import GroundingDINOTool, OCRTool, ToolExecutor
        print("✓ tools imported")
    except Exception as e:
        print(f"✗ tools import failed: {e}")
        return False
    
    try:
        from inference import VQAInference
        print("✓ inference imported")
    except Exception as e:
        print(f"✗ inference import failed: {e}")
        return False
    
    print("\n✓ All imports successful!")
    return True


def test_config_values():
    """Test configuration values."""
    print("\nTesting configuration values...")
    
    import config
    
    assert config.DEFAULT_DEPTH == 3, "DEFAULT_DEPTH should be 3"
    print(f"✓ DEFAULT_DEPTH = {config.DEFAULT_DEPTH}")
    
    assert config.CLIP_TOP_K_TOKENS == 10, "CLIP_TOP_K_TOKENS should be 10"
    print(f"✓ CLIP_TOP_K_TOKENS = {config.CLIP_TOP_K_TOKENS}")
    
    assert config.LLM_MODEL_NAME is not None, "LLM_MODEL_NAME should be set"
    print(f"✓ LLM_MODEL_NAME = {config.LLM_MODEL_NAME}")
    
    assert config.CLIP_MODEL_NAME is not None, "CLIP_MODEL_NAME should be set"
    print(f"✓ CLIP_MODEL_NAME = {config.CLIP_MODEL_NAME}")
    
    print("\n✓ All configuration tests passed!")
    return True


def test_tool_executor_interface():
    """Test tool executor interface."""
    print("\nTesting tool executor interface...")
    
    from tools import ToolExecutor
    from PIL import Image
    import numpy as np
    
    # Create dummy image
    dummy_image = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
    
    tool_executor = ToolExecutor()
    print("✓ ToolExecutor initialized")
    
    # Test grounding_dino tool call parsing
    result = tool_executor.execute_tool_call(dummy_image, 'grounding_dino(query="person")')
    assert result is not None
    print(f"✓ Grounding DINO tool call executed: {result[:50]}...")
    
    # Test ocr tool call parsing
    result = tool_executor.execute_tool_call(dummy_image, 'ocr()')
    assert result is not None
    print(f"✓ OCR tool call executed: {result[:50]}...")
    
    print("\n✓ All tool executor tests passed!")
    return True


def test_prompts_format():
    """Test that prompt templates can be formatted."""
    print("\nTesting prompt templates...")
    
    import config
    
    # Test atomicity check prompt
    prompt = config.ATOMICITY_CHECK_PROMPT.format(
        context="test context",
        question="test question"
    )
    assert "test context" in prompt
    assert "test question" in prompt
    print("✓ ATOMICITY_CHECK_PROMPT formatting works")
    
    # Test tool call generation prompt
    prompt = config.TOOL_CALL_GENERATION_PROMPT.format(
        context="test context",
        question="test question"
    )
    assert "test context" in prompt
    assert "test question" in prompt
    print("✓ TOOL_CALL_GENERATION_PROMPT formatting works")
    
    # Test sub-question generation prompt
    prompt = config.SUB_QUESTION_GENERATION_PROMPT.format(
        context="test context",
        question="test question"
    )
    assert "test context" in prompt
    assert "test question" in prompt
    print("✓ SUB_QUESTION_GENERATION_PROMPT formatting works")
    
    # Test aggregate results prompt
    prompt = config.AGGREGATE_RESULTS_PROMPT.format(
        question="test question",
        context="test context",
        sub_results="test results"
    )
    assert "test question" in prompt
    assert "test context" in prompt
    assert "test results" in prompt
    print("✓ AGGREGATE_RESULTS_PROMPT formatting works")
    
    print("\n✓ All prompt template tests passed!")
    return True


def main():
    """Run all tests."""
    print("="*80)
    print("VQAv2 Inference System - Structure Tests")
    print("="*80)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
    
    # Test config
    if not test_config_values():
        all_passed = False
    
    # Test tool executor
    if not test_tool_executor_interface():
        all_passed = False
    
    # Test prompts
    if not test_prompts_format():
        all_passed = False
    
    print("\n" + "="*80)
    if all_passed:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed!")
    print("="*80)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
