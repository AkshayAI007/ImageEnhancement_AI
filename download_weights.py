#!/usr/bin/env python3
"""
Download RealESRGAN weights using wget/curl or requests
"""

import os
import subprocess
import sys
from pathlib import Path
import requests

def download_with_wget_curl():
    """Download using wget or curl"""
    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)
    
    url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    output_path = weights_dir / "RealESRGAN_x4plus.pth"
    
    # Try wget first
    try:
        result = subprocess.run(['wget', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("🔧 Using wget to download weights...")
            cmd = ['wget', url, '-O', str(output_path), '--progress=bar']
            
            result = subprocess.run(cmd)
            if result.returncode == 0:
                print("✅ Download successful with wget!")
                return True
    except FileNotFoundError:
        print("⚠️ wget not found, trying curl...")
    
    # Try curl
    try:
        result = subprocess.run(['curl', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("🔧 Using curl to download weights...")
            cmd = ['curl', '-L', url, '-o', str(output_path), '--progress-bar']
            
            result = subprocess.run(cmd)
            if result.returncode == 0:
                print("✅ Download successful with curl!")
                return True
    except FileNotFoundError:
        print("⚠️ curl not found, using Python requests...")
    
    # Fallback to requests
    print("🐍 Using Python requests as fallback...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        print(f"📥 Downloading RealESRGAN_x4plus.pth ({total_size / (1024*1024):.1f} MB)...")
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rProgress: {percent:.1f}%", end="", flush=True)
        
        print(f"\n✅ Download successful with requests!")
        return True
        
    except Exception as e:
        print(f"❌ All download methods failed: {e}")
        return False

def verify_weights():
    """Verify that weights are properly downloaded"""
    weights_path = Path("weights/RealESRGAN_x4plus.pth")
    
    if not weights_path.exists():
        print("❌ Weights file not found")
        return False
    
    file_size = weights_path.stat().st_size
    size_mb = file_size / (1024 * 1024)
    
    print(f"📊 File size: {size_mb:.1f} MB")
    
    if size_mb < 50:
        print("❌ File seems too small (should be ~67MB)")
        return False
    
    print("✅ Weights file looks good!")
    print("🎉 Ready to run your application!")
    return True

def main():
    print("🎯 Downloading RealESRGAN Weights")
    print("=" * 50)
    
    # Check if weights already exist
    weights_path = Path("weights/RealESRGAN_x4plus.pth")
    if weights_path.exists():
        print("✅ Weights already exist!")
        if verify_weights():
            return
        else:
            print("🔄 Re-downloading weights...")
    
    # Download weights
    success = download_with_wget_curl()
    
    if success:
        print("\n🔍 Verifying weights...")
        verify_weights()
    else:
        print("\n❌ Download failed!")
        print("💡 Try downloading manually from:")
        print("   https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth")

if __name__ == "__main__":
    main()
