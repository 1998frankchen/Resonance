#!/usr/bin/env python3
"""Create sample images for demo purposes."""

import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def create_sample_image(text, filename, size=(224, 224)):
    """Create a sample image with text for demo purposes."""
    # Create a new image with white background
    img = Image.new('RGB', size, color='white')
    draw = ImageDraw.Draw(img)
    
    # Try to use a default font, fallback to default if not available
    try:
        font = ImageFont.load_default()
    except:
        font = None
    
    # Calculate text position to center it
    if font:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    else:
        text_width, text_height = draw.textsize(text)
    
    x = (size[0] - text_width) // 2
    y = (size[1] - text_height) // 2
    
    # Draw text
    draw.text((x, y), text, fill='black', font=font)
    
    # Add some color elements
    draw.rectangle([10, 10, 50, 50], fill='red', outline='black')
    draw.ellipse([size[0]-60, 10, size[0]-10, 60], fill='blue', outline='black')
    
    return img

def create_demo_images():
    """Create a set of demo images."""
    images_dir = "/root/work/Resonance/demos/images"
    os.makedirs(images_dir, exist_ok=True)
    
    sample_texts = [
        "Cat sitting on chair",
        "Dog playing in park", 
        "Beautiful sunset",
        "City skyline at night",
        "Mountain landscape",
        "Ocean waves",
        "Forest path",
        "Flower garden",
        "Birds flying",
        "Butterfly on flower"
    ]
    
    for i, text in enumerate(sample_texts):
        img = create_sample_image(text, f"demo_{i+1:02d}.jpg")
        img.save(os.path.join(images_dir, f"demo_{i+1:02d}.jpg"))
        print(f"Created {text} -> demo_{i+1:02d}.jpg")

if __name__ == "__main__":
    create_demo_images()
    print("Demo images created successfully!")