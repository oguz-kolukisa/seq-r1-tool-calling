"""Tests for VQAv2 inference system."""

import os
import sys


def test_imports():
    """Test module imports."""
    print("Testing imports...")
    try:
        import config
        from clip_extractor import CLIPContextExtractor
        from llm_wrapper import LLMWrapper
        from tools import GroundingDINOTool, OCRTool, ToolExecutor
        from inference import VQAInference
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_config():
    """Test configuration."""
    print("\nTesting configuration...")
    try:
        import config
        assert config.DEFAULT_DEPTH == 3
        assert config.CLIP_TOP_K_TOKENS == 10
        assert config.LLM_MODEL_NAME is not None
        assert config.CLIP_MODEL_NAME is not None
        print("✓ Configuration valid")
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


def test_file_structure():
    """Test required files exist."""
    print("\nTesting file structure...")
    required_files = [
        'config.py', 'inference.py', 'llm_wrapper.py',
        'clip_extractor.py', 'tools.py', 'run_inference.py',
        'requirements.txt', 'README.md'
    ]
    
    missing = [f for f in required_files if not os.path.exists(f)]
    
    if missing:
        print(f"✗ Missing files: {missing}")
        return False
    
    print(f"✓ All {len(required_files)} required files present")
    return True


def test_tool_executor():
    """Test tool executor interface."""
    print("\nTesting tool executor...")
    try:
        from tools import ToolExecutor
        from PIL import Image
        import numpy as np
        
        dummy_image = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        executor = ToolExecutor()
        
        # Test parameter extraction
        param = executor._extract_parameter('grounding_dino(query="person")', 'query')
        assert param == "person"
        
        print("✓ Tool executor works")
        return True
    except Exception as e:
        print(f"✗ Tool executor test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("VQAv2 Inference System Tests")
    print("="*60)
    
    tests = [
        test_file_structure,
        test_imports,
        test_config,
        test_tool_executor,
    ]
    
    results = [test() for test in tests]
    passed = sum(results)
    total = len(results)
    
    print("\n" + "="*60)
    print(f"Results: {passed}/{total} tests passed")
    print("="*60)
    
    return all(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
