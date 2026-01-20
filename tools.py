"""Tool wrappers for Grounding DINO and OCR."""

from PIL import Image
from typing import List, Dict, Any
import torch
import numpy as np
import config


class GroundingDINOTool:
    """Grounding DINO tool for object detection and grounding tasks."""
    
    def __init__(self, config_dict: Dict[str, Any] = None):
        """Initialize Grounding DINO tool.
        
        Args:
            config_dict: Configuration dictionary for the tool
        """
        self.config = config_dict or config.GROUNDING_DINO_CONFIG
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def _load_model(self):
        """Lazy load Grounding DINO model."""
        if self.model is not None:
            return
            
        try:
            from groundingdino.util.inference import load_model
            
            print(f"Loading Grounding DINO model...")
            self.model = load_model(
                self.config["config_file"],
                self.config["checkpoint"]
            )
            self.model = self.model.to(self.device)
            print("Grounding DINO model loaded successfully!")
        except Exception as e:
            print(f"Warning: Could not load Grounding DINO model: {e}")
            print("Using fallback mode.")
            self.model = "fallback"
        
    def __call__(self, image: Image.Image, query: str) -> Dict[str, Any]:
        """Perform grounding task on the image.
        
        Args:
            image: PIL Image object
            query: Text query for grounding (e.g., "person", "car")
            
        Returns:
            Dictionary containing detected objects and their bounding boxes
        """
        self._load_model()
        
        if self.model == "fallback":
            # Fallback when model is not available
            return {
                "query": query,
                "detections": [],
                "count": 0,
                "message": f"Fallback mode: Grounding DINO not available for query '{query}'"
            }
        
        try:
            from groundingdino.util.inference import predict
            
            # Run inference
            boxes, logits, phrases = predict(
                model=self.model,
                image=image,
                caption=query,
                box_threshold=self.config["box_threshold"],
                text_threshold=self.config["text_threshold"]
            )
            
            # Convert to standard format
            detections = []
            h, w = image.size[1], image.size[0]
            
            for box, score, phrase in zip(boxes, logits, phrases):
                # Convert normalized coordinates to pixel coordinates
                x1, y1, x2, y2 = box.cpu().numpy()
                x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
                
                detections.append({
                    "box": [x1, y1, x2, y2],
                    "score": float(score),
                    "label": phrase
                })
            
            return {
                "query": query,
                "detections": detections,
                "count": len(detections),
                "message": f"Detected {len(detections)} instances of '{query}'"
            }
            
        except Exception as e:
            return {
                "query": query,
                "detections": [],
                "count": 0,
                "error": str(e),
                "message": f"Error during detection: {e}"
            }


class OCRTool:
    """OCR tool for text recognition tasks using EasyOCR."""
    
    def __init__(self, config_dict: Dict[str, Any] = None):
        """Initialize OCR tool.
        
        Args:
            config_dict: Configuration dictionary for the tool
        """
        self.config = config_dict or config.OCR_CONFIG
        self.reader = None
        
    def _load_model(self):
        """Lazy load EasyOCR reader."""
        if self.reader is not None:
            return
            
        try:
            import easyocr
            
            print(f"Loading EasyOCR reader...")
            self.reader = easyocr.Reader(
                self.config["languages"],
                gpu=self.config["gpu"]
            )
            print("EasyOCR reader loaded successfully!")
        except Exception as e:
            print(f"Warning: Could not load EasyOCR: {e}")
            print("Using fallback mode.")
            self.reader = "fallback"
        
    def __call__(self, image: Image.Image) -> Dict[str, Any]:
        """Perform OCR on the image.
        
        Args:
            image: PIL Image object
            
        Returns:
            Dictionary containing detected text and locations
        """
        self._load_model()
        
        if self.reader == "fallback":
            # Fallback when EasyOCR is not available
            return {
                "text_detections": [],
                "full_text": "",
                "count": 0,
                "message": "Fallback mode: EasyOCR not available"
            }
        
        try:
            # Convert PIL image to numpy array
            image_np = np.array(image)
            
            # Perform OCR
            results = self.reader.readtext(image_np)
            
            # Format results
            text_detections = []
            all_text = []
            
            for bbox, text, confidence in results:
                # bbox is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                x1, y1 = min(x_coords), min(y_coords)
                x2, y2 = max(x_coords), max(y_coords)
                
                text_detections.append({
                    "text": text,
                    "box": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": float(confidence)
                })
                all_text.append(text)
            
            full_text = " ".join(all_text)
            
            return {
                "text_detections": text_detections,
                "full_text": full_text,
                "count": len(text_detections),
                "message": f"Detected {len(text_detections)} text regions"
            }
            
        except Exception as e:
            return {
                "text_detections": [],
                "full_text": "",
                "count": 0,
                "error": str(e),
                "message": f"Error during OCR: {e}"
            }


class ToolExecutor:
    """Executor for managing and calling different tools."""
    
    def __init__(self):
        """Initialize tool executor with available tools."""
        self.grounding_dino = GroundingDINOTool()
        self.ocr = OCRTool()
        
    def execute_tool_call(self, image: Image.Image, tool_call_str: str) -> str:
        """Parse and execute a tool call string.
        
        Args:
            image: PIL Image object
            tool_call_str: Tool call string (e.g., "grounding_dino(query='person')")
            
        Returns:
            String representation of tool execution result
        """
        tool_call_str = tool_call_str.strip()
        
        # Parse tool call
        if tool_call_str.startswith("grounding_dino("):
            # Extract query parameter
            query = self._extract_parameter(tool_call_str, "query")
            result = self.grounding_dino(image, query)
            
            # Format result for LLM
            if result["count"] > 0:
                return f"Detected {result['count']} instances of '{query}': {result['detections']}"
            else:
                return f"No instances of '{query}' detected"
            
        elif tool_call_str.startswith("ocr()"):
            result = self.ocr(image)
            
            # Format result for LLM
            if result["count"] > 0:
                return f"OCR detected {result['count']} text regions. Full text: '{result['full_text']}'"
            else:
                return "No text detected in the image"
            
        else:
            return f"Unknown tool call: {tool_call_str}"
    
    def _extract_parameter(self, tool_call_str: str, param_name: str) -> str:
        """Extract parameter value from tool call string.
        
        Args:
            tool_call_str: Tool call string
            param_name: Name of parameter to extract
            
        Returns:
            Parameter value as string
        """
        import re
        pattern = f'{param_name}=["\']([^"\']+)["\']'
        match = re.search(pattern, tool_call_str)
        if match:
            return match.group(1)
        return ""
