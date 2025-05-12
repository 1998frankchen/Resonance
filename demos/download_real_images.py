#!/usr/bin/env python3
"""Download real images for Resonance demos from the internet.

This script downloads a variety of real images from publicly available sources
to create authentic demo datasets for vision-language model training.
"""

import os
import requests
from PIL import Image
import io
from pathlib import Path

def download_and_process_image(url, filename, target_size=(224, 224)):
    """Download an image from URL and process it for demo use.
    
    Args:
        url: URL to download image from
        filename: Local filename to save as
        target_size: Tuple of (width, height) to resize to
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"Downloading {filename}...")
        
        # Download image
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Open and process image
        image = Image.open(io.BytesIO(response.content))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize while maintaining aspect ratio
        image.thumbnail(target_size, Image.Resampling.LANCZOS)
        
        # Create a square canvas and paste image in center
        canvas = Image.new('RGB', target_size, (255, 255, 255))
        offset_x = (target_size[0] - image.width) // 2
        offset_y = (target_size[1] - image.height) // 2
        canvas.paste(image, (offset_x, offset_y))
        
        # Save image
        output_path = f"demos/images/{filename}"
        canvas.save(output_path, 'JPEG', quality=85)
        print(f"✅ Saved {filename}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to download {filename}: {e}")
        return False

def download_demo_images():
    """Download a curated set of demo images."""
    
    # Create images directory
    os.makedirs("demos/images", exist_ok=True)
    
    # Curated list of publicly available images from various sources
    # Using images from Unsplash and other public domain sources
    image_urls = {
        "demo_01.jpg": "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=500&h=500&fit=crop", # Cat
        "demo_02.jpg": "https://images.unsplash.com/photo-1552053831-71594a27632d?w=500&h=500&fit=crop", # Dog in park
        "demo_03.jpg": "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=500&h=500&fit=crop", # Sunset
        "demo_04.jpg": "https://images.unsplash.com/photo-1519501025264-65ba15a82390?w=500&h=500&fit=crop", # City skyline
        "demo_05.jpg": "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=500&h=500&fit=crop", # Mountains
        "demo_06.jpg": "https://images.unsplash.com/photo-1544551763-46a013bb70d5?w=500&h=500&fit=crop", # Ocean waves
        "demo_07.jpg": "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=500&h=500&fit=crop", # Forest path
        "demo_08.jpg": "https://images.unsplash.com/photo-1416879595882-3373a0480b5b?w=500&h=500&fit=crop", # Flower garden
        "demo_09.jpg": "https://images.unsplash.com/photo-1426604966848-d7adac402bff?w=500&h=500&fit=crop", # Birds flying
        "demo_10.jpg": "https://images.unsplash.com/photo-1444927714506-8492d94b5ba0?w=500&h=500&fit=crop", # Butterfly on flower
    }
    
    successful_downloads = 0
    
    for filename, url in image_urls.items():
        if download_and_process_image(url, filename):
            successful_downloads += 1
    
    print(f"\n🎉 Successfully downloaded {successful_downloads}/{len(image_urls)} demo images!")
    
    if successful_downloads < len(image_urls):
        print("⚠️ Some downloads failed. You may need to manually replace failed images.")
    
    return successful_downloads > 0

if __name__ == "__main__":
    print("📷 Downloading real images for Resonance demos...")
    print("=" * 50)
    
    success = download_demo_images()
    
    if success:
        print("\n✅ Demo images ready for authentic vision-language training!")
        print("Check demos/images/ directory for the downloaded images.")
    else:
        print("\n❌ Failed to download demo images. Check your internet connection.")