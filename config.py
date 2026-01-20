"""Configuration file for VQAv2 inference with recursive tool calling."""

# Default recursion depth for answer function
DEFAULT_DEPTH = 3

# Model configurations
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
LLM_MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"  # Can be changed to any HuggingFace model

# CLIP configuration
CLIP_TOP_K_TOKENS = 10

# Tool configurations
GROUNDING_DINO_CONFIG = {
    "model_name": "IDEA-Research/grounding-dino-base",
    "box_threshold": 0.35,
    "text_threshold": 0.25,
}

OCR_CONFIG = {
    "tool": "easyocr",  # Can be changed to other OCR tools
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

AGGREGATE_RESULTS_PROMPT = """Given the following question, sub-questions, and their answers, synthesize a final answer to the original question.

Original Question: {question}
Image Context: {context}

Sub-questions and Answers:
{sub_results}

Provide a comprehensive final answer to the original question.
"""
