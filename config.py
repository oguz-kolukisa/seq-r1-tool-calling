"""Configuration file for VQAv2 inference with recursive tool calling."""

# Default recursion depth for answer function
DEFAULT_DEPTH = 3

# Model configurations
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
LLM_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"  # Qwen 3B model

# CLIP configuration
CLIP_TOP_K_TOKENS = 10

# Tool configurations
GROUNDING_DINO_CONFIG = {
    "config_file": "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    "checkpoint": "groundingdino_swint_ogc.pth",
    "box_threshold": 0.35,
    "text_threshold": 0.25,
}

OCR_CONFIG = {
    "languages": ["en"],  # EasyOCR languages
    "gpu": True,  # Use GPU if available
}

# VQAv2 Dataset configuration
VQAV2_CONFIG = {
    "data_dir": "data/vqav2",
    "coco_dir": "data/coco",
    "download_splits": ["train", "val"],  # Which splits to download
}

# Prompt templates
ATOMICITY_CHECK_PROMPT = """Given the following question and image context, determine if the question is atomic (can be answered with a single tool call) or needs to be broken down into sub-questions.

Image Context: {context}
Question: {question}

Respond with either "ATOMIC" or "NOT_ATOMIC" followed by a brief explanation.
"""

TOOL_CALL_GENERATION_PROMPT = """Given the following atomic question and image context, generate the appropriate tool call to answer it.

Image Context: {context}
Question: {question}

Available tools:
- grounding_dino: For object detection and grounding tasks. Usage: grounding_dino(query="object to detect")
- ocr: For text recognition tasks. Usage: ocr()

Generate the tool call in the format: tool_name(parameters)
"""

SUB_QUESTION_GENERATION_PROMPT = """Given the following complex question and image context, break it down into smaller atomic sub-questions that can be answered independently.

Image Context: {context}
Question: {question}

Generate a list of sub-questions, one per line, that together will help answer the original question.
"""

AGGREGATE_RESULTS_PROMPT = """Given the following question, tool results or sub-question answers, reason about them and provide a final answer to the original question.

Original Question: {question}
Image Context: {context}

Tool Results / Sub-questions and Answers:
{sub_results}

Based on the information above, provide a clear and direct answer to the original question.
"""
