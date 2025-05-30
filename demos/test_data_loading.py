#!/usr/bin/env python3
"""Test data loading for Resonance demo datasets.

This script verifies that the demo datasets can be loaded correctly
by the actual Resonance training code.
"""

import sys
import os
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_dataset_format():
    """Test that demo datasets have correct format."""
    print("🧪 Testing demo dataset formats...")
    
    # Test SFT dataset
    with open("demos/data/demo_sft.json", "r") as f:
        sft_data = json.load(f)
    
    # Validate SFT format
    for i, sample in enumerate(sft_data):
        assert "image" in sample, f"SFT sample {i} missing 'image' field"
        assert "conversations" in sample, f"SFT sample {i} missing 'conversations' field"
        
        conversations = sample["conversations"]
        assert len(conversations) >= 2, f"SFT sample {i} needs at least 2 conversation turns"
        assert conversations[0]["from"] == "user", f"SFT sample {i} first turn should be from user"
        assert conversations[1]["from"] == "assistant", f"SFT sample {i} second turn should be from assistant"
        
        # Check image exists
        img_path = f"demos/images/{sample['image']}"
        assert os.path.exists(img_path), f"Image {img_path} not found for SFT sample {i}"
    
    print(f"✅ SFT dataset: {len(sft_data)} samples validated")
    
    # Test DPO dataset  
    with open("demos/data/demo_dpo.json", "r") as f:
        dpo_data = json.load(f)
        
    # Validate DPO format
    for i, sample in enumerate(dpo_data):
        required_fields = ["image", "prompt", "chosen", "rejected"]
        for field in required_fields:
            assert field in sample, f"DPO sample {i} missing '{field}' field"
        
        # Validate chosen is better than rejected
        assert len(sample["chosen"]) > len(sample["rejected"]), \
            f"DPO sample {i}: chosen response should be longer than rejected"
            
        # Check image exists
        img_path = f"demos/images/{sample['image']}"
        assert os.path.exists(img_path), f"Image {img_path} not found for DPO sample {i}"
    
    print(f"✅ DPO dataset: {len(dpo_data)} samples validated")

def test_image_files():
    """Test that all demo images are valid."""
    print("🖼️ Testing demo images...")
    
    from PIL import Image
    
    image_dir = Path("demos/images")
    image_files = list(image_dir.glob("demo_*.jpg"))
    
    assert len(image_files) >= 5, f"Expected at least 5 demo images, found {len(image_files)}"
    
    for img_file in image_files:
        try:
            with Image.open(img_file) as img:
                # Basic validation
                assert img.size[0] > 0 and img.size[1] > 0, f"Invalid image dimensions: {img.size}"
                assert img.mode in ["RGB", "RGBA"], f"Invalid image mode: {img.mode}"
        except Exception as e:
            raise AssertionError(f"Failed to load image {img_file}: {e}")
    
    print(f"✅ Images: {len(image_files)} files validated")

def test_training_script_args():
    """Test that demo training scripts have correct arguments."""
    print("📝 Testing demo training scripts...")
    
    scripts_dir = Path("demos/scripts")
    
    # Check demo_sft.sh
    with open(scripts_dir / "demo_sft.sh", "r") as f:
        sft_content = f.read()
    
    # Verify key arguments are present
    assert "src/resonance/sft.py" in sft_content, "SFT script should reference correct Python module"
    assert "demos/data/demo_sft.json" in sft_content, "SFT script should reference demo dataset"
    assert "vlquery_json" in sft_content, "SFT script should use correct dataset format"
    
    # Check demo_dpo.sh
    with open(scripts_dir / "demo_dpo.sh", "r") as f:
        dpo_content = f.read()
    
    assert "src/resonance/dpo.py" in dpo_content, "DPO script should reference correct Python module"
    assert "demos/data/demo_dpo.json" in dpo_content, "DPO script should reference demo dataset"
    assert "plain_dpo" in dpo_content, "DPO script should use correct dataset format"
    
    print("✅ Demo scripts: Arguments validated")

def main():
    """Run all demo tests."""
    print("🚀 Running Resonance Demo Tests")
    print("=" * 40)
    
    try:
        test_dataset_format()
        test_image_files() 
        test_training_script_args()
        
        print("=" * 40)
        print("🎉 All demo tests passed! The demos are ready to use.")
        print("\nNext steps:")
        print("1. Download a pre-trained model (e.g., Qwen-VL-Chat)")
        print("2. Replace <path-to-pretrained-model> in demo scripts")
        print("3. Run: bash demos/scripts/demo_sft.sh")
        
    except Exception as e:
        print(f"❌ Demo test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()