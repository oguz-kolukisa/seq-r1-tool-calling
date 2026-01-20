"""HTML Report Generator for VQA Inference Results."""

import os
import base64
from io import BytesIO
from datetime import datetime
from typing import List, Dict, Any, Optional
from PIL import Image


class InferenceReport:
    """Tracks and generates HTML reports for VQA inference."""
    
    def __init__(self):
        """Initialize the report."""
        self.steps = []
        self.image_path = None
        self.question = None
        self.final_answer = None
        self.start_time = None
        self.end_time = None
        
    def set_input(self, image_path: str, question: str):
        """Set the input image and question."""
        self.image_path = image_path
        self.question = question
        self.start_time = datetime.now()
        
    def add_step(self, step_data: Dict[str, Any]):
        """Add an inference step to the report."""
        self.steps.append(step_data)
        
    def set_final_answer(self, answer: str):
        """Set the final answer."""
        self.final_answer = answer
        self.end_time = datetime.now()
        
    def _image_to_base64(self, image_path: str) -> str:
        """Convert image to base64 string for embedding in HTML."""
        try:
            with Image.open(image_path) as img:
                # Resize if too large
                max_size = (800, 800)
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
                
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                return f"data:image/png;base64,{img_str}"
        except Exception as e:
            return f"Error loading image: {e}"
    
    def _format_step_html(self, step: Dict[str, Any], index: int) -> str:
        """Format a single step as HTML."""
        depth = step.get('depth', 0)
        indent = "  " * depth
        step_type = step.get('type', 'unknown')
        
        html = f'<div class="step depth-{depth}" id="step-{index}">\n'
        html += f'  <div class="step-header">\n'
        html += f'    <span class="step-number">Step {index + 1}</span>\n'
        html += f'    <span class="step-depth">Depth: {depth}</span>\n'
        html += f'    <span class="step-type">{step_type.replace("_", " ").title()}</span>\n'
        html += f'  </div>\n'
        html += f'  <div class="step-content">\n'
        
        # Question
        if 'question' in step:
            html += f'    <div class="field"><strong>Question:</strong> {step["question"]}</div>\n'
        
        # CLIP Context
        if 'context' in step:
            html += f'    <div class="field"><strong>Visual Context (CLIP):</strong> <code>{step["context"]}</code></div>\n'
        
        # Atomicity
        if 'is_atomic' in step:
            atomic_class = 'atomic' if step['is_atomic'] else 'not-atomic'
            atomic_text = 'ATOMIC' if step['is_atomic'] else 'NOT ATOMIC'
            html += f'    <div class="field"><strong>Atomicity:</strong> <span class="badge {atomic_class}">{atomic_text}</span></div>\n'
        
        # Tool Calls
        if 'tool_calls' in step:
            html += f'    <div class="field"><strong>Tool Calls:</strong></div>\n'
            html += f'    <ul class="tool-calls">\n'
            for tool_call in step['tool_calls']:
                html += f'      <li><code>{tool_call}</code></li>\n'
            html += f'    </ul>\n'
        
        # Tool Results
        if 'tool_results' in step:
            html += f'    <div class="field"><strong>Tool Results:</strong></div>\n'
            html += f'    <div class="tool-results">\n'
            for tool_call, result in step['tool_results']:
                html += f'      <div class="tool-result">\n'
                html += f'        <div class="tool-call"><code>{tool_call}</code></div>\n'
                html += f'        <div class="result-text">{result}</div>\n'
                html += f'      </div>\n'
            html += f'    </div>\n'
        
        # Sub-questions
        if 'sub_questions' in step:
            html += f'    <div class="field"><strong>Sub-questions:</strong></div>\n'
            html += f'    <ol class="sub-questions">\n'
            for sub_q in step['sub_questions']:
                html += f'      <li>{sub_q}</li>\n'
            html += f'    </ol>\n'
        
        # Sub-results
        if 'sub_results' in step:
            html += f'    <div class="field"><strong>Sub-question Results:</strong></div>\n'
            html += f'    <div class="sub-results">\n'
            for sub_q, sub_a in step['sub_results']:
                html += f'      <div class="sub-result">\n'
                html += f'        <div class="sub-question"><strong>Q:</strong> {sub_q}</div>\n'
                html += f'        <div class="sub-answer"><strong>A:</strong> {sub_a}</div>\n'
                html += f'      </div>\n'
            html += f'    </div>\n'
        
        # Answer
        if 'answer' in step:
            html += f'    <div class="field answer-field"><strong>Answer:</strong> {step["answer"]}</div>\n'
        
        html += f'  </div>\n'
        html += f'</div>\n'
        
        return html
    
    def generate_html(self, output_path: str):
        """Generate HTML report and save to file."""
        
        # Calculate duration
        duration = ""
        if self.start_time and self.end_time:
            delta = self.end_time - self.start_time
            duration = f"{delta.total_seconds():.2f} seconds"
        
        # Convert image to base64
        image_data = self._image_to_base64(self.image_path) if self.image_path else ""
        
        # Generate steps HTML
        steps_html = ""
        for i, step in enumerate(self.steps):
            steps_html += self._format_step_html(step, i)
        
        # HTML template
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VQA Inference Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .header {{
            text-align: center;
            padding-bottom: 30px;
            border-bottom: 3px solid #4CAF50;
            margin-bottom: 30px;
        }}
        
        .header h1 {{
            color: #2196F3;
            margin-bottom: 10px;
            font-size: 2.5em;
        }}
        
        .header .timestamp {{
            color: #666;
            font-size: 0.9em;
        }}
        
        .input-section {{
            background: #f9f9f9;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            border-left: 4px solid #2196F3;
        }}
        
        .input-section h2 {{
            color: #2196F3;
            margin-bottom: 15px;
        }}
        
        .image-container {{
            text-align: center;
            margin: 20px 0;
        }}
        
        .image-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            border: 2px solid #ddd;
        }}
        
        .question-text {{
            font-size: 1.2em;
            font-weight: bold;
            color: #333;
            padding: 15px;
            background: white;
            border-radius: 5px;
            margin: 10px 0;
        }}
        
        .final-answer-section {{
            background: #e8f5e9;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            border-left: 4px solid #4CAF50;
        }}
        
        .final-answer-section h2 {{
            color: #4CAF50;
            margin-bottom: 15px;
        }}
        
        .final-answer-text {{
            font-size: 1.2em;
            color: #2e7d32;
            font-weight: 500;
        }}
        
        .execution-info {{
            display: flex;
            justify-content: space-between;
            padding: 15px;
            background: #fff3e0;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        
        .execution-info .info-item {{
            text-align: center;
        }}
        
        .execution-info .info-label {{
            font-size: 0.9em;
            color: #666;
        }}
        
        .execution-info .info-value {{
            font-size: 1.3em;
            font-weight: bold;
            color: #f57c00;
        }}
        
        .steps-section h2 {{
            color: #333;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #ddd;
        }}
        
        .step {{
            background: white;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            transition: all 0.3s ease;
        }}
        
        .step:hover {{
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            border-color: #2196F3;
        }}
        
        .step.depth-0 {{
            border-left: 5px solid #2196F3;
        }}
        
        .step.depth-1 {{
            border-left: 5px solid #4CAF50;
            margin-left: 30px;
        }}
        
        .step.depth-2 {{
            border-left: 5px solid #FF9800;
            margin-left: 60px;
        }}
        
        .step-header {{
            display: flex;
            gap: 15px;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #e0e0e0;
        }}
        
        .step-number {{
            background: #2196F3;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
        }}
        
        .step-depth {{
            background: #757575;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
        }}
        
        .step-type {{
            background: #e0e0e0;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
            color: #333;
        }}
        
        .step-content {{
            padding: 10px 0;
        }}
        
        .field {{
            margin-bottom: 15px;
        }}
        
        .field strong {{
            color: #555;
            display: block;
            margin-bottom: 5px;
        }}
        
        .badge {{
            padding: 5px 12px;
            border-radius: 5px;
            font-weight: bold;
            font-size: 0.9em;
        }}
        
        .badge.atomic {{
            background: #c8e6c9;
            color: #2e7d32;
        }}
        
        .badge.not-atomic {{
            background: #ffccbc;
            color: #d84315;
        }}
        
        code {{
            background: #f5f5f5;
            padding: 3px 8px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            color: #d32f2f;
        }}
        
        .tool-calls, .sub-questions {{
            margin: 10px 0;
            padding-left: 20px;
        }}
        
        .tool-calls li, .sub-questions li {{
            margin: 5px 0;
        }}
        
        .tool-results {{
            margin: 10px 0;
        }}
        
        .tool-result {{
            background: #f9f9f9;
            padding: 12px;
            border-radius: 5px;
            margin: 8px 0;
            border-left: 3px solid #2196F3;
        }}
        
        .tool-call {{
            font-weight: bold;
            margin-bottom: 5px;
        }}
        
        .result-text {{
            color: #555;
            padding-left: 10px;
        }}
        
        .sub-results {{
            margin: 10px 0;
        }}
        
        .sub-result {{
            background: #f0f7ff;
            padding: 12px;
            border-radius: 5px;
            margin: 8px 0;
            border-left: 3px solid #4CAF50;
        }}
        
        .sub-question {{
            margin-bottom: 5px;
            color: #1976d2;
        }}
        
        .sub-answer {{
            color: #2e7d32;
            padding-left: 10px;
        }}
        
        .answer-field {{
            background: #e8f5e9;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #4CAF50;
        }}
        
        .footer {{
            text-align: center;
            padding-top: 30px;
            margin-top: 30px;
            border-top: 2px solid #e0e0e0;
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 VQA Inference Report</h1>
            <div class="timestamp">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
        </div>
        
        <div class="input-section">
            <h2>📥 Input</h2>
            <div class="image-container">
                <img src="{image_data}" alt="Input Image">
            </div>
            <div class="question-text">
                <strong>Question:</strong> {self.question}
            </div>
        </div>
        
        <div class="final-answer-section">
            <h2>✅ Final Answer</h2>
            <div class="final-answer-text">
                {self.final_answer}
            </div>
        </div>
        
        <div class="execution-info">
            <div class="info-item">
                <div class="info-label">Total Steps</div>
                <div class="info-value">{len(self.steps)}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Execution Time</div>
                <div class="info-value">{duration}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Max Depth</div>
                <div class="info-value">{max([s.get('depth', 0) for s in self.steps]) if self.steps else 0}</div>
            </div>
        </div>
        
        <div class="steps-section">
            <h2>📊 Inference Steps</h2>
            {steps_html}
        </div>
        
        <div class="footer">
            <p>VQA Recursive Inference System with LLM-driven Tool Decomposition</p>
            <p>Powered by Qwen 3B, CLIP, Grounding DINO, and OCR</p>
        </div>
    </div>
</body>
</html>"""
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"HTML report generated: {output_path}")
        return output_path


class ReportingVQAInference:
    """VQA Inference wrapper that generates HTML reports."""
    
    def __init__(self, base_inference):
        """Initialize with base inference engine.
        
        Args:
            base_inference: VQAInference instance
        """
        self.base_inference = base_inference
        self.report = None
        
    def answer(
        self,
        image: Image.Image,
        question: str,
        depth: int = 0
    ) -> str:
        """Recursive answer function that tracks steps for reporting."""
        
        # Initialize step data
        step_data = {
            'depth': depth,
            'question': question,
            'type': 'answer_step'
        }
        
        # Check depth limit
        if depth >= self.base_inference.max_depth:
            step_data['type'] = 'max_depth_reached'
            step_data['answer'] = f"Maximum recursion depth ({self.base_inference.max_depth}) reached."
            self.report.add_step(step_data)
            return step_data['answer']
        
        # Extract visual context
        context = self.base_inference.clip_extractor.extract_context(image, question)
        step_data['context'] = context
        
        # Check atomicity
        is_atomic = self.base_inference.llm.check_atomicity(question, context)
        step_data['is_atomic'] = is_atomic
        
        if is_atomic:
            step_data['type'] = 'atomic_question'
            
            # Generate and execute tool calls
            tool_calls = self.base_inference.llm.generate_tool_call(question, context)
            step_data['tool_calls'] = tool_calls
            
            tool_results = []
            for tool_call in tool_calls:
                tool_result = self.base_inference.tool_executor.execute_tool_call(image, tool_call)
                tool_results.append((tool_call, tool_result))
            
            step_data['tool_results'] = tool_results
            
            # Reason about results
            final_answer = self.base_inference.llm.aggregate_results(
                question, context, tool_results
            )
            step_data['answer'] = final_answer
            
            self.report.add_step(step_data)
            return final_answer
        
        else:
            step_data['type'] = 'complex_question'
            
            # Generate sub-questions
            sub_questions = self.base_inference.llm.generate_sub_questions(question, context)
            step_data['sub_questions'] = sub_questions
            
            # Recursively answer sub-questions
            sub_results = []
            for sub_q in sub_questions:
                sub_answer = self.answer(image, sub_q, depth + 1)
                sub_results.append((sub_q, sub_answer))
            
            step_data['sub_results'] = sub_results
            
            # Aggregate results
            final_answer = self.base_inference.llm.aggregate_results(question, context, sub_results)
            step_data['answer'] = final_answer
            
            self.report.add_step(step_data)
            return final_answer
    
    def inference_vqav2_with_report(
        self,
        image_path: str,
        question: str,
        report_path: str = "inference_report.html"
    ) -> dict:
        """Run inference and generate HTML report.
        
        Args:
            image_path: Path to the image file
            question: Question about the image
            report_path: Path where HTML report should be saved
            
        Returns:
            Dictionary containing question, answer, and report path
        """
        print(f"\n{'='*80}")
        print(f"Processing VQAv2 Question with Report Generation")
        print(f"Question: {question}")
        print(f"Image: {image_path}")
        print(f"{'='*80}")
        
        # Initialize report
        self.report = InferenceReport()
        self.report.set_input(image_path, question)
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Run inference with reporting
        answer = self.answer(image, question, depth=0)
        
        # Set final answer
        self.report.set_final_answer(answer)
        
        # Generate HTML report
        self.report.generate_html(report_path)
        
        print(f"\n{'='*80}")
        print(f"Inference Complete!")
        print(f"Answer: {answer}")
        print(f"Report saved to: {report_path}")
        print(f"{'='*80}\n")
        
        return {
            'question': question,
            'answer': answer,
            'image_path': image_path,
            'report_path': report_path
        }
