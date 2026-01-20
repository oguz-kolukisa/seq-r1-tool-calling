"""Standalone HTML report demo without requiring model dependencies."""

import base64
from io import BytesIO
from datetime import datetime


def create_demo_html_report():
    """Create a demo HTML report showing the structure."""
    
    # Create a simple demo image (1x1 red pixel as base64)
    demo_image_base64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VQA Inference Report - Demo</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
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
            padding: 20px;
            background: #e3f2fd;
            border-radius: 8px;
        }}
        
        .image-container img {{
            max-width: 600px;
            height: auto;
            border-radius: 8px;
            border: 2px solid #2196F3;
        }}
        
        .image-placeholder {{
            padding: 100px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-size: 1.5em;
            border-radius: 8px;
            text-align: center;
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
            justify-content: space-around;
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
            font-size: 1.5em;
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
        
        .step.depth-0 {{ border-left: 5px solid #2196F3; }}
        .step.depth-1 {{ border-left: 5px solid #4CAF50; margin-left: 30px; }}
        .step.depth-2 {{ border-left: 5px solid #FF9800; margin-left: 60px; }}
        
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
                <div class="image-placeholder">
                    🖼️ Sample Image:<br>
                    Street scene with car, people, and stop sign
                </div>
            </div>
            <div class="question-text">
                <strong>Question:</strong> How many people are standing near the red car?
            </div>
        </div>
        
        <div class="final-answer-section">
            <h2>✅ Final Answer</h2>
            <div class="final-answer-text">
                There are 2 people standing near the red car.
            </div>
        </div>
        
        <div class="execution-info">
            <div class="info-item">
                <div class="info-label">Total Steps</div>
                <div class="info-value">7</div>
            </div>
            <div class="info-item">
                <div class="info-label">Execution Time</div>
                <div class="info-value">2.35s</div>
            </div>
            <div class="info-item">
                <div class="info-label">Max Depth</div>
                <div class="info-value">2</div>
            </div>
        </div>
        
        <div class="steps-section">
            <h2>📊 Inference Steps</h2>
            
            <!-- Step 1: Complex Question at Depth 0 -->
            <div class="step depth-0">
                <div class="step-header">
                    <span class="step-number">Step 1</span>
                    <span class="step-depth">Depth: 0</span>
                    <span class="step-type">Complex Question</span>
                </div>
                <div class="step-content">
                    <div class="field"><strong>Question:</strong> How many people are standing near the red car?</div>
                    <div class="field"><strong>Visual Context (CLIP):</strong> <code>car, red, person, street, vehicle, people, outdoor, road, stop sign, building</code></div>
                    <div class="field"><strong>Atomicity:</strong> <span class="badge not-atomic">NOT ATOMIC</span></div>
                    <div class="field"><strong>Sub-questions:</strong></div>
                    <ol class="sub-questions">
                        <li>Where is the red car located?</li>
                        <li>Where are people in the image?</li>
                        <li>How many people are near the car?</li>
                    </ol>
                </div>
            </div>
            
            <!-- Step 2: First Sub-question at Depth 1 -->
            <div class="step depth-1">
                <div class="step-header">
                    <span class="step-number">Step 2</span>
                    <span class="step-depth">Depth: 1</span>
                    <span class="step-type">Atomic Question</span>
                </div>
                <div class="step-content">
                    <div class="field"><strong>Question:</strong> Where is the red car located?</div>
                    <div class="field"><strong>Visual Context (CLIP):</strong> <code>car, red, vehicle, outdoor, road, street, parking, automobile, sedan, transportation</code></div>
                    <div class="field"><strong>Atomicity:</strong> <span class="badge atomic">ATOMIC</span></div>
                    <div class="field"><strong>Tool Calls:</strong></div>
                    <ul class="tool-calls">
                        <li><code>grounding_dino(query="red car")</code></li>
                    </ul>
                    <div class="field"><strong>Tool Results:</strong></div>
                    <div class="tool-results">
                        <div class="tool-result">
                            <div class="tool-call"><code>grounding_dino(query="red car")</code></div>
                            <div class="result-text">Detected 1 object: "red car" at bounding box [120, 200, 380, 450] with confidence 0.95</div>
                        </div>
                    </div>
                    <div class="field answer-field"><strong>Answer:</strong> The red car is located in the center-left area of the image, positioned on the street.</div>
                </div>
            </div>
            
            <!-- Step 3: Second Sub-question at Depth 1 -->
            <div class="step depth-1">
                <div class="step-header">
                    <span class="step-number">Step 3</span>
                    <span class="step-depth">Depth: 1</span>
                    <span class="step-type">Atomic Question</span>
                </div>
                <div class="step-content">
                    <div class="field"><strong>Question:</strong> Where are people in the image?</div>
                    <div class="field"><strong>Visual Context (CLIP):</strong> <code>person, people, human, pedestrian, man, woman, standing, walking, outdoor, street</code></div>
                    <div class="field"><strong>Atomicity:</strong> <span class="badge atomic">ATOMIC</span></div>
                    <div class="field"><strong>Tool Calls:</strong></div>
                    <ul class="tool-calls">
                        <li><code>grounding_dino(query="person")</code></li>
                    </ul>
                    <div class="field"><strong>Tool Results:</strong></div>
                    <div class="tool-results">
                        <div class="tool-result">
                            <div class="tool-call"><code>grounding_dino(query="person")</code></div>
                            <div class="result-text">Detected 2 objects: "person" at [480, 320, 520, 380] (conf: 0.92), "person" at [530, 320, 570, 380] (conf: 0.88)</div>
                        </div>
                    </div>
                    <div class="field answer-field"><strong>Answer:</strong> There are 2 people standing in the right portion of the image.</div>
                </div>
            </div>
            
            <!-- Step 4: Third Sub-question at Depth 1 -->
            <div class="step depth-1">
                <div class="step-header">
                    <span class="step-number">Step 4</span>
                    <span class="step-depth">Depth: 1</span>
                    <span class="step-type">Atomic Question</span>
                </div>
                <div class="step-content">
                    <div class="field"><strong>Question:</strong> How many people are near the car?</div>
                    <div class="field"><strong>Visual Context (CLIP):</strong> <code>person, car, near, close, proximity, standing, beside, next to, people, vehicle</code></div>
                    <div class="field"><strong>Atomicity:</strong> <span class="badge atomic">ATOMIC</span></div>
                    <div class="field"><strong>Tool Calls:</strong></div>
                    <ul class="tool-calls">
                        <li><code>grounding_dino(query="person near car")</code></li>
                    </ul>
                    <div class="field"><strong>Tool Results:</strong></div>
                    <div class="tool-results">
                        <div class="tool-result">
                            <div class="tool-call"><code>grounding_dino(query="person near car")</code></div>
                            <div class="result-text">Detected 2 people near the car based on spatial proximity analysis.</div>
                        </div>
                    </div>
                    <div class="field answer-field"><strong>Answer:</strong> There are 2 people near the car.</div>
                </div>
            </div>
            
            <!-- Step 5: Aggregation at Depth 0 -->
            <div class="step depth-0">
                <div class="step-header">
                    <span class="step-number">Step 5</span>
                    <span class="step-depth">Depth: 0</span>
                    <span class="step-type">Result Aggregation</span>
                </div>
                <div class="step-content">
                    <div class="field"><strong>Sub-question Results:</strong></div>
                    <div class="sub-results">
                        <div class="sub-result">
                            <div class="sub-question"><strong>Q:</strong> Where is the red car located?</div>
                            <div class="sub-answer"><strong>A:</strong> The red car is located in the center-left area of the image, positioned on the street.</div>
                        </div>
                        <div class="sub-result">
                            <div class="sub-question"><strong>Q:</strong> Where are people in the image?</div>
                            <div class="sub-answer"><strong>A:</strong> There are 2 people standing in the right portion of the image.</div>
                        </div>
                        <div class="sub-result">
                            <div class="sub-question"><strong>Q:</strong> How many people are near the car?</div>
                            <div class="sub-answer"><strong>A:</strong> There are 2 people near the car.</div>
                        </div>
                    </div>
                    <div class="field answer-field"><strong>Answer:</strong> There are 2 people standing near the red car.</div>
                </div>
            </div>
            
        </div>
        
        <div class="footer">
            <p><strong>VQA Recursive Inference System with LLM-driven Tool Decomposition</strong></p>
            <p>Powered by Qwen 3B, CLIP, Grounding DINO, and OCR</p>
            <p style="margin-top: 10px; font-size: 0.85em;">
                This report shows the complete inference process including:<br>
                • Visual context extraction with CLIP<br>
                • Atomicity determination by LLM<br>
                • Tool call generation and execution<br>
                • Recursive sub-question decomposition<br>
                • Result aggregation with LLM reasoning
            </p>
        </div>
    </div>
</body>
</html>"""
    
    return html


if __name__ == "__main__":
    print("=" * 80)
    print("Generating Demo HTML Report")
    print("=" * 80)
    
    html_content = create_demo_html_report()
    output_path = "demo_inference_report.html"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n✅ Demo HTML report generated: {output_path}")
    print("\nThis demo shows the structure and features of the VQA inference report:")
    print("  • Input image visualization")
    print("  • Question and final answer")
    print("  • Execution metrics (steps, time, depth)")
    print("  • Step-by-step inference process with:")
    print("    - CLIP visual context")
    print("    - Atomicity checks")
    print("    - Tool calls and results")
    print("    - Sub-question generation")
    print("    - LLM reasoning")
    print("  • Color-coded recursion depth")
    print("  • Interactive hover effects")
    print("\nOpen the HTML file in your browser to view the full report!")
    print("=" * 80)
