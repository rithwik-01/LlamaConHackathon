"""
Image utilities for LORE.

This module provides functions for handling and processing images 
for multimodal analysis with LLMs.
"""
import base64
import os
from io import BytesIO
from typing import Optional, Dict, Any, List
import logging

from PIL import Image

logger = logging.getLogger(__name__)

def encode_image_to_base64(image_path: str) -> Optional[str]:
    """
    Encode an image to base64.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Base64 encoded image or None if encoding fails
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error encoding image {image_path}: {e}")
        return None

def encode_pil_image_to_base64(image: Image.Image) -> str:
    """
    Encode a PIL Image to base64.
    
    Args:
        image: PIL Image object
        
    Returns:
        Base64 encoded image
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def resize_image_if_needed(image_path: str, max_pixels: int = 4 * 1024 * 1024) -> Optional[str]:
    """
    Resize an image if it exceeds the maximum number of pixels.
    
    Args:
        image_path: Path to image file
        max_pixels: Maximum number of pixels
        
    Returns:
        Path to resized image (might be the same as input if no resizing needed)
    """
    try:
        img = Image.open(image_path)
        width, height = img.size
        pixels = width * height
        
        if pixels <= max_pixels:
            return image_path
        
        # Calculate new dimensions
        ratio = (max_pixels / pixels) ** 0.5
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        
        # Resize image
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Save resized image
        dirname = os.path.dirname(image_path)
        basename = os.path.basename(image_path)
        name, ext = os.path.splitext(basename)
        resized_path = os.path.join(dirname, f"{name}_resized{ext}")
        
        resized_img.save(resized_path)
        logger.info(f"Resized image from {width}x{height} to {new_width}x{new_height}")
        
        return resized_path
    except Exception as e:
        logger.error(f"Error resizing image {image_path}: {e}")
        return image_path

def format_image_for_llama(image_path: str) -> List[Dict[str, Any]]:
    """
    Format an image for use with Llama API.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Formatted content list for Llama API
    """
    # Resize image if needed to meet API requirements
    resized_path = resize_image_if_needed(image_path)
    if not resized_path:
        return []
    
    # Encode image to base64
    base64_image = encode_image_to_base64(resized_path)
    if not base64_image:
        return []
    
    # Format as Llama API expects
    return [
        {
            "type": "text",
            "text": "I've uploaded a design diagram for the codebase. Please analyze it and describe what you see, including the architecture, components, and relationships."
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        }
    ]
