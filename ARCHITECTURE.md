# System Architecture

## Overview

This document describes the architecture of the VQAv2 Inference System with Recursive Tool Calling.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     VQAInference (Main)                         │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │           answer(image, question, depth)                   │ │
│  │                                                            │ │
│  │  1. Extract Context (CLIP)                                │ │
│  │         ↓                                                  │ │
│  │  2. Check Atomicity (LLM)                                 │ │
│  │         ↓                                                  │ │
│  │     ┌───┴───┐                                             │ │
│  │     │       │                                             │ │
│  │  ATOMIC   NOT ATOMIC                                      │ │
│  │     │       │                                             │ │
│  │     │       └──→ 3b. Generate Sub-Questions (LLM)        │ │
│  │     │                    ↓                                │ │
│  │     │            4. Recursive answer() calls             │ │
│  │     │                    ↓                                │ │
│  │     │            5. Aggregate Results (LLM) ────┐        │ │
│  │     │                                            │        │ │
│  │     └──→ 3a. Generate Tool Call (LLM)           │        │ │
│  │              ↓                                   │        │ │
│  │          Execute Tool                            │        │ │
│  │              ↓                                   │        │ │
│  │          [Return Result] ←───────────────────────┘        │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘

        ┌──────────────────────────────────────────────────┐
        │              Supporting Components                │
        └──────────────────────────────────────────────────┘

┌─────────────────┐  ┌──────────────────┐  ┌────────────────┐
│ CLIPExtractor   │  │   LLMWrapper     │  │  ToolExecutor  │
│                 │  │                  │  │                │
│ • Top-10 tokens │  │ • Atomicity      │  │ • Grounding    │
│ • Visual        │  │   Check          │  │   DINO         │
│   concepts      │  │ • Tool Call Gen  │  │ • OCR          │
│                 │  │ • Sub-Q Gen      │  │                │
│                 │  │ • Aggregation    │  │                │
└─────────────────┘  └──────────────────┘  └────────────────┘
```

## Component Details

### 1. VQAInference (Main Orchestrator)

**Purpose**: Main entry point and orchestrator for the VQA inference pipeline.

**Key Methods**:
- `__init__()`: Initialize all components (CLIP, LLM, Tools)
- `answer()`: Recursive function implementing the core logic
- `inference_vqav2()`: High-level interface for single question-image pair
- `batch_inference()`: Process multiple samples

**Recursion Control**:
- Maximum depth limit (default: 3)
- Depth tracking and enforcement
- Prevents infinite recursion

### 2. CLIPContextExtractor

**Purpose**: Extract visual context from images since the LLM cannot see images directly.

**Process**:
1. Load image with PIL
2. Process image through CLIP vision encoder
3. Compare against candidate visual concepts
4. Extract top-10 most likely concepts
5. Format as context string for LLM

**Output Example**:
```
"Visual context: person, indoor scene, table, chair, window, room, 
 clothes, furniture, many, building"
```

### 3. LLMWrapper

**Purpose**: Wrap HuggingFace language model for different generation tasks.

**Key Methods**:

#### a. `check_atomicity(question, context) -> bool`
- Determines if question is atomic or needs decomposition
- Uses ATOMICITY_CHECK_PROMPT template
- Returns True if atomic, False otherwise

**Example**:
```python
# Atomic question
"What color is the car?" → True

# Non-atomic question  
"How many people are wearing red shirts?" → False
```

#### b. `generate_tool_call(question, context) -> str`
- Generates appropriate tool call for atomic questions
- Uses TOOL_CALL_GENERATION_PROMPT template
- Returns tool call string

**Example**:
```python
"What text is on the sign?" → 'ocr()'
"Where is the person?" → 'grounding_dino(query="person")'
```

#### c. `generate_sub_questions(question, context) -> List[str]`
- Breaks complex question into atomic sub-questions
- Uses SUB_QUESTION_GENERATION_PROMPT template
- Returns list of sub-questions

**Example**:
```python
"How many people are wearing red shirts?"
→ [
    "Where are the people in the image?",
    "What color shirts are people wearing?", 
    "How many people are wearing red shirts?"
  ]
```

#### d. `aggregate_results(question, context, sub_results) -> str`
- Combines answers from sub-questions
- Uses AGGREGATE_RESULTS_PROMPT template
- Returns final answer

**Example**:
```python
Sub-results: [
    ("Where are people?", "3 people detected"),
    ("What shirt colors?", "2 red, 1 blue"),
]
→ "There are 2 people wearing red shirts"
```

### 4. ToolExecutor

**Purpose**: Manage and execute different tools.

**Available Tools**:

#### a. GroundingDINOTool
- Object detection and grounding
- Finds objects and their locations
- Returns bounding boxes and confidence scores

**Usage**:
```python
grounding_dino(query="person")
grounding_dino(query="red car")
```

#### b. OCRTool
- Text recognition
- Extracts text from images
- Returns detected text and locations

**Usage**:
```python
ocr()
```

**Tool Call Execution**:
1. Parse tool call string
2. Extract parameters
3. Call appropriate tool
4. Format and return results

## Data Flow

### Example: Complex Question Processing

**Input**:
- Image: Photo of people at a table
- Question: "How many people are sitting and what are they doing?"

**Processing Flow**:

```
1. answer(image, question, depth=0)
   │
   ├→ CLIP extracts context: "person, people, table, indoor, sitting, ..."
   │
   ├→ LLM checks atomicity: NOT_ATOMIC (needs multiple observations)
   │
   └→ LLM generates sub-questions:
       ["How many people are in the image?",
        "Are the people sitting or standing?",
        "What objects are near the people?"]

2. For each sub-question, answer(image, sub_q, depth=1):
   
   2a. answer(image, "How many people are in the image?", depth=1)
       ├→ CLIP context: "person, people, ..."
       ├→ LLM: ATOMIC
       └→ Tool call: grounding_dino(query="person")
           → Result: "3 people detected"
   
   2b. answer(image, "Are the people sitting or standing?", depth=1)
       ├→ CLIP context: "sitting, chair, ..."
       ├→ LLM: ATOMIC
       └→ Tool call: grounding_dino(query="sitting person")
           → Result: "3 sitting people detected"
   
   2c. answer(image, "What objects are near the people?", depth=1)
       ├→ CLIP context: "table, chair, ..."
       ├→ LLM: ATOMIC
       └→ Tool call: grounding_dino(query="table")
           → Result: "1 table detected near people"

3. Aggregate results:
   LLM combines sub-answers:
   → "There are 3 people sitting at a table"
```

## Recursion Control

### Depth Limiting

```python
def answer(image, question, depth=0):
    if depth >= max_depth:
        return "Max depth reached"
    
    # ... rest of logic
    
    if not atomic:
        for sub_q in sub_questions:
            answer(image, sub_q, depth + 1)  # Increment depth
```

### Maximum Depth Examples

**depth=1**: Only one level of decomposition
- Limited to simple decompositions
- Fast but less thorough

**depth=2**: Two levels of decomposition  
- Balanced approach
- Handles moderately complex questions

**depth=3** (default): Three levels of decomposition
- Most thorough
- Can handle complex nested questions
- Slower but more accurate

**depth > 3**: Deep recursion
- Rarely needed
- Risk of circular decomposition
- Very slow

## Prompt Engineering

All LLM functions use carefully designed prompts:

### 1. Atomicity Check Prompt
- Provides context and question
- Asks for binary decision
- Requests brief explanation

### 2. Tool Call Generation Prompt
- Lists available tools
- Shows usage examples
- Requests specific format

### 3. Sub-Question Generation Prompt
- Explains decomposition goal
- Requests atomic sub-questions
- One per line format

### 4. Aggregate Results Prompt
- Provides all sub-results
- Requests coherent synthesis
- Emphasizes completeness

## Model Requirements

### LLM
- Must be causal language model
- Should support instruction following
- Recommended: 7B+ parameters
- Examples: Llama-2, Mistral, Phi-2

### CLIP
- Standard CLIP architecture
- Pre-trained on image-text pairs
- Examples: openai/clip-vit-base-patch32

## Configuration

All settings in `config.py`:

```python
# Recursion
DEFAULT_DEPTH = 3

# Models
LLM_MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# CLIP settings
CLIP_TOP_K_TOKENS = 10

# Tool settings
GROUNDING_DINO_CONFIG = {...}
OCR_CONFIG = {...}

# Prompts
ATOMICITY_CHECK_PROMPT = "..."
TOOL_CALL_GENERATION_PROMPT = "..."
SUB_QUESTION_GENERATION_PROMPT = "..."
AGGREGATE_RESULTS_PROMPT = "..."
```

## Error Handling

### Depth Limit Exceeded
- Returns default message
- Logs warning
- Prevents stack overflow

### Tool Execution Failure
- Returns error message
- Continues processing
- Logged for debugging

### LLM Generation Failure
- Returns error indication
- Can retry with different parameters
- Falls back to simple answer

## Performance Considerations

### Optimization Strategies

1. **Caching**:
   - Cache CLIP contexts for repeated images
   - Cache atomicity checks for similar questions

2. **Batching**:
   - Process multiple questions in parallel
   - Batch LLM generation calls

3. **Early Stopping**:
   - Stop recursion when confident
   - Skip unnecessary decomposition

4. **Model Size**:
   - Trade-off between accuracy and speed
   - Smaller models for faster inference

### Typical Inference Times

(With GPU, estimates):
- Simple atomic question: 2-5 seconds
- 1-level decomposition: 5-10 seconds
- 2-level decomposition: 10-20 seconds
- 3-level decomposition: 20-40 seconds

## Future Enhancements

1. **Dynamic Depth**: Automatically adjust depth based on question complexity
2. **Parallel Execution**: Process sub-questions in parallel
3. **Confidence Scoring**: Track and report confidence levels
4. **Memory**: Remember previous questions for context
5. **Multi-modal LLM**: Use vision-language models directly
6. **Tool Learning**: Learn which tools to use from examples
7. **Error Recovery**: Better handling of tool failures
