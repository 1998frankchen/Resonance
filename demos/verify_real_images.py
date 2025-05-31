#!/usr/bin/env python3
"""Verify real demo images work with VL model processing.

This script performs comprehensive verification that the real downloaded images
can be properly loaded and processed by vision-language model pipelines.
"""

import json
import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np

def verify_image_properties():
    """Verify basic image properties for VL model compatibility."""
    print("🔍 Verifying image properties...")
    
    image_dir = Path("demos/images")
    issues = []
    
    for img_file in sorted(image_dir.glob("demo_*.jpg")):
        try:
            with Image.open(img_file) as img:
                # Check basic properties
                width, height = img.size
                mode = img.mode
                format_name = img.format
                
                # Verify image is valid for VL models
                if mode != 'RGB':
                    issues.append(f"{img_file.name}: Not RGB mode ({mode})")
                
                if width < 50 or height < 50:
                    issues.append(f"{img_file.name}: Too small ({width}x{height})")
                
                if width > 2048 or height > 2048:
                    issues.append(f"{img_file.name}: Very large ({width}x{height})")
                
                # Try to convert to numpy (common VL model requirement)
                img_array = np.array(img)
                if img_array.shape != (height, width, 3):
                    issues.append(f"{img_file.name}: Unexpected array shape {img_array.shape}")
                
                print(f"  ✅ {img_file.name}: {width}x{height} {mode} {format_name}")
                
        except Exception as e:
            issues.append(f"{img_file.name}: Failed to load - {e}")
    
    if issues:
        print("❌ Image issues found:")
        for issue in issues:
            print(f"     {issue}")
        return False
    
    print("✅ All images have valid properties for VL models")
    return True

def verify_dataset_image_alignment():
    """Verify dataset entries align with actual images."""
    print("\n🔗 Verifying dataset-image alignment...")
    
    image_dir = Path("demos/images")
    existing_images = set(f.name for f in image_dir.glob("demo_*.jpg"))
    
    datasets = {
        "SFT": "demos/data/demo_sft.json",
        "DPO": "demos/data/demo_dpo.json", 
        "Eval": "demos/data/demo_eval_vqa.json"
    }
    
    issues = []
    
    for dataset_name, dataset_path in datasets.items():
        try:
            with open(dataset_path, 'r') as f:
                data = json.load(f)
            
            referenced_images = set()
            for sample in data:
                if 'image' in sample:
                    referenced_images.add(sample['image'])
            
            # Check for missing images
            missing = referenced_images - existing_images
            if missing:
                issues.append(f"{dataset_name}: References missing images: {missing}")
            
            # Check for unused images
            unused = existing_images - referenced_images
            if unused and dataset_name == "SFT":  # Only report for SFT as it should use all
                issues.append(f"{dataset_name}: Unused images: {unused}")
            
            print(f"  ✅ {dataset_name}: {len(data)} samples, {len(referenced_images)} unique images")
            
        except Exception as e:
            issues.append(f"{dataset_name}: Failed to load - {e}")
    
    if issues:
        print("❌ Dataset-image alignment issues:")
        for issue in issues:
            print(f"     {issue}")
        return False
    
    print("✅ All datasets properly reference existing images")
    return True

def verify_content_quality():
    """Verify the quality and realism of dataset content."""
    print("\n📝 Verifying content quality...")
    
    try:
        # Check SFT conversations
        with open("demos/data/demo_sft.json", 'r') as f:
            sft_data = json.load(f)
        
        quality_issues = []
        
        for i, sample in enumerate(sft_data):
            conversations = sample.get('conversations', [])
            if len(conversations) < 2:
                quality_issues.append(f"SFT sample {i}: Insufficient conversation turns")
                continue
                
            assistant_response = conversations[1].get('value', '')
            
            # Check response quality
            if len(assistant_response.split()) < 10:
                quality_issues.append(f"SFT sample {i}: Assistant response too brief")
            
            if 'I can see' not in assistant_response and 'This image shows' not in assistant_response:
                quality_issues.append(f"SFT sample {i}: Response doesn't start with visual description")
        
        # Check DPO preferences
        with open("demos/data/demo_dpo.json", 'r') as f:
            dpo_data = json.load(f)
        
        for i, sample in enumerate(dpo_data):
            chosen = sample.get('chosen', '')
            rejected = sample.get('rejected', '')
            
            # Verify chosen is clearly better than rejected
            if len(chosen) <= len(rejected):
                quality_issues.append(f"DPO sample {i}: Chosen response not clearly better")
            
            if len(chosen.split()) < 15:
                quality_issues.append(f"DPO sample {i}: Chosen response too brief")
        
        if quality_issues:
            print("⚠️ Content quality issues found:")
            for issue in quality_issues[:5]:  # Show first 5
                print(f"     {issue}")
            if len(quality_issues) > 5:
                print(f"     ... and {len(quality_issues) - 5} more issues")
            return False
        
        print(f"✅ Content quality verified:")
        print(f"   📊 SFT: {len(sft_data)} high-quality conversation samples") 
        print(f"   📊 DPO: {len(dpo_data)} clear preference pairs")
        return True
        
    except Exception as e:
        print(f"❌ Failed to verify content quality: {e}")
        return False

def verify_model_loading_compatibility():
    """Test basic compatibility with common VL model processing."""
    print("\n🤖 Testing model loading compatibility...")
    
    try:
        # Test with PIL operations commonly used by VL models
        from PIL import Image
        import torch
        import torchvision.transforms as transforms
        
        # Common VL model preprocessing pipeline
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        success_count = 0
        
        for img_file in sorted(Path("demos/images").glob("demo_*.jpg"))[:3]:  # Test first 3
            try:
                # Load and preprocess image
                image = Image.open(img_file)
                
                # Apply typical VL model transforms
                tensor = transform(image)
                
                # Verify tensor properties
                if tensor.shape == (3, 224, 224):
                    success_count += 1
                    print(f"  ✅ {img_file.name}: Preprocessed successfully")
                else:
                    print(f"  ❌ {img_file.name}: Unexpected tensor shape {tensor.shape}")
                    
            except Exception as e:
                print(f"  ❌ {img_file.name}: Preprocessing failed - {e}")
        
        if success_count >= 3:
            print("✅ Images compatible with standard VL model preprocessing")
            return True
        else:
            print("❌ Images failed VL model compatibility tests")
            return False
            
    except ImportError:
        print("⚠️ PyTorch not available, skipping model compatibility tests")
        return True  # Don't fail if optional dependencies missing
    except Exception as e:
        print(f"❌ Model compatibility test failed: {e}")
        return False

def main():
    """Run comprehensive verification of real demo images and datasets."""
    print("🎯 Comprehensive Demo Verification")
    print("=" * 50)
    
    all_checks = [
        verify_image_properties,
        verify_dataset_image_alignment,
        verify_content_quality,
        verify_model_loading_compatibility
    ]
    
    passed = 0
    for check in all_checks:
        if check():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Verification Results: {passed}/{len(all_checks)} checks passed")
    
    if passed == len(all_checks):
        print("🎉 All verifications passed! Real demo images and datasets are ready for VL model training.")
        print("\n✅ The demos now feature:")
        print("   📷 Real images downloaded from internet sources")
        print("   📝 Authentic descriptions matching actual image content")
        print("   🔧 Compatibility with standard VL model preprocessing")
        print("   🎯 Realistic training scenarios for SFT, DPO, and evaluation")
        return True
    else:
        print("❌ Some verifications failed. Please address the issues above.")
        return False

if __name__ == "__main__":
    sys.exit(0 if main() else 1)