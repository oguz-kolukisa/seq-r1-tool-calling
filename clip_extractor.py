"""CLIP model wrapper for extracting image context."""

import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from typing import List
import config


class CLIPContextExtractor:
    """Extracts visual context from images using CLIP."""
    
    def __init__(self, model_name: str = config.CLIP_MODEL_NAME):
        """Initialize CLIP model and processor.
        
        Args:
            model_name: HuggingFace model name for CLIP
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.top_k = config.CLIP_TOP_K_TOKENS
        
    def extract_context(self, image: Image.Image, question: str = None) -> str:
        """Extract top-k most likely tokens/concepts from the image.
        
        Args:
            image: PIL Image object
            question: Optional question for conditioning
            
        Returns:
            String representation of image context (top-k concepts)
        """
        # Define a set of common visual concepts for zero-shot classification
        candidate_concepts = [
            "person", "people", "man", "woman", "child", "object",
            "building", "car", "animal", "dog", "cat", "food",
            "indoor scene", "outdoor scene", "nature", "city",
            "text", "sign", "number", "color", "red", "blue",
            "green", "yellow", "large", "small", "many", "few",
            "table", "chair", "book", "phone", "computer", "sky",
            "tree", "grass", "road", "room", "window", "door",
            "clothes", "furniture", "vehicle", "sport", "game",
            "kitchen", "bedroom", "street", "beach", "mountain"
        ]
        
        # Process image and text
        inputs = self.processor(
            text=[f"a photo of {concept}" for concept in candidate_concepts],
            images=image,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
        
        # Get top-k concepts
        top_k_probs, top_k_indices = torch.topk(probs[0], self.top_k)
        top_concepts = [candidate_concepts[idx] for idx in top_k_indices.cpu()]
        
        # Format as context string
        context = "Visual context: " + ", ".join(top_concepts)
        return context
