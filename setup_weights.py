#!/usr/bin/env python3
"""
Alternative methods to download RealESRGAN weights
"""

import os
import requests
import sys
from pathlib import Path
import subprocess

def method_1_manual_download():
    """Method 1: Manual download instructions"""
    print("🔗 Method 1: Manual Download (Recommended)")
    print("=" * 50)
    print("1. Open this link in your browser:")
    print("   https://drive.google.com/file/d/1zpbQcgM9YreBN9n7IMINPDaFJHM2nxF9/view?usp=drive_link")
    print("2. Click the download button")
    print("3. Save the file to your Downloads folder")
    print("4. Run this script again and select Method 4 to move the file")
    print()

def method_2_gdown():
    """Method 2: Using gdown library"""
    print("📦 Method 2: Using gdown library")
    print("=" * 50)
    
    try:
        import gdown
        print("✅ gdown is already installed")
    except ImportError:
        print("Installing gdown...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown
    
    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)
    
    # Your Google Drive file ID
    file_id = "1zpbQcgM9YreBN9n7IMINPDaFJHM2nxF9"
    output_path = weights_dir / "RealESRGAN_x4plus.pth"
    
    print(f"📥 Downloading to: {output_path}")
    
    try:
        # Method 2a: Direct download
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, str(output_path), quiet=False)
        
        if output_path.exists() and output_path.stat().st_size > 1000000:
            print("✅ Download successful!")
            return True
        else:
            print("❌ Download failed or file too small")
            return False
            
    except Exception as e:
        print(f"❌ gdown method failed: {e}")
        return False

def method_3_official_weights():
    """Method 3: Download official weights from GitHub"""
    print("🏛️ Method 3: Official RealESRGAN weights from GitHub")
    print("=" * 50)
    
    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)
    
    # Official RealESRGAN weights URLs
    official_urls = [
        {
            "name": "RealESRGAN_x4plus.pth",
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            "size": "~67MB"
        },
        {
            "name": "RealESRGAN_x2plus.pth", 
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x2plus.pth",
            "size": "~67MB"
        }
    ]
    
    print("Available official weights:")
    for i, weight in enumerate(official_urls, 1):
        print(f"{i}. {weight['name']} ({weight['size']})")
    
    choice = input("Choose which to download (1 or 2): ").strip()
    
    if choice in ['1', '2']:
        selected = official_urls[int(choice) - 1]
        output_path = weights_dir / selected["name"]
        
        print(f"📥 Downloading {selected['name']}...")
        
        try:
            response = requests.get(selected["url"], stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"\rProgress: {percent:.1f}%", end="", flush=True)
            
            print(f"\n✅ Downloaded successfully: {output_path}")
            print(f"📊 File size: {output_path.stat().st_size / (1024*1024):.1f} MB")
            return True
            
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return False
    else:
        print("❌ Invalid choice")
        return False

def method_4_move_downloaded_file():
    """Method 4: Move manually downloaded file"""
    print("📁 Method 4: Move manually downloaded file")
    print("=" * 50)
    
    # Common download locations
    possible_locations = [
        Path.home() / "Downloads",
        Path.home() / "Desktop",
        Path(".")
    ]
    
    # Look for downloaded files
    found_files = []
    for location in possible_locations:
        if location.exists():
            for pattern in ["*.pth", "*realsr*", "*RealESR*", "*real*"]:
                found_files.extend(location.glob(pattern))
    
    if found_files:
        print("🔍 Found these potential weight files:")
        for i, file_path in enumerate(found_files, 1):
            size_mb = file_path.stat().st_size / (1024*1024)
            print(f"{i}. {file_path} ({size_mb:.1f} MB)")
        
        choice = input("Choose which file to use (enter number): ").strip()
        
        try:
            selected_file = found_files[int(choice) - 1]
            
            # Create weights directory
            weights_dir = Path("weights")
            weights_dir.mkdir(exist_ok=True)
            
            # Copy file to weights directory
            import shutil
            destination = weights_dir / "RealESRGAN_x4plus.pth"
            shutil.copy2(selected_file, destination)
            
            print(f"✅ Copied {selected_file} to {destination}")
            return True
            
        except (ValueError, IndexError):
            print("❌ Invalid choice")
            return False
        except Exception as e:
            print(f"❌ Error copying file: {e}")
            return False
    else:
        print("❌ No potential weight files found")
        print("💡 Please download the file manually first")
        return False

def method_5_wget_curl():
    """Method 5: Using wget or curl"""
    print("🌐 Method 5: Using wget or curl")
    print("=" * 50)
    
    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)
    
    # Try wget first
    try:
        result = subprocess.run(['wget', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("🔧 Using wget...")
            cmd = [
                'wget', 
                'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
                '-O', str(weights_dir / "RealESRGAN_x4plus.pth")
            ]
            
            result = subprocess.run(cmd)
            if result.returncode == 0:
                print("✅ Download successful with wget!")
                return True
    except FileNotFoundError:
        pass
    
    # Try curl
    try:
        result = subprocess.run(['curl', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("🔧 Using curl...")
            cmd = [
                'curl', '-L',
                'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
                '-o', str(weights_dir / "RealESRGAN_x4plus.pth")
            ]
            
            result = subprocess.run(cmd)
            if result.returncode == 0:
                print("✅ Download successful with curl!")
                return True
    except FileNotFoundError:
        pass
    
    print("❌ Neither wget nor curl is available")
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
    
    # Try to test with RealESRGAN
    try:
        from realesrgan import RealESRGAN
        import torch
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = RealESRGAN(device, scale=4)
        model.load_weights(str(weights_path))
        
        print("✅ Weights loaded successfully!")
        print("🎉 Ready to run your Flask app!")
        return True
        
    except ImportError:
        print("⚠️ Cannot test weights (RealESRGAN not installed)")
        print("💡 Install with: pip install realesrgan torch")
        return True  # File exists and has correct size
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        return False

def main():
    print("🎯 RealESRGAN Weights Download Helper")
    print("=" * 50)
    
    methods = [
        ("Manual Download (Recommended)", method_1_manual_download),
        ("Using gdown library", method_2_gdown),
        ("Official GitHub weights", method_3_official_weights),
        ("Move downloaded file", method_4_move_downloaded_file),
        ("Using wget/curl", method_5_wget_curl)
    ]
    
    print("Available download methods:")
    for i, (name, _) in enumerate(methods, 1):
        print(f"{i}. {name}")
    
    choice = input("\nChoose a method (1-5): ").strip()
    
    try:
        method_index = int(choice) - 1
        if 0 <= method_index < len(methods):
            name, method_func = methods[method_index]
            print(f"\n🚀 Using: {name}")
            print("=" * 50)
            
            if method_index == 0:  # Manual download
                method_func()
                return
            
            success = method_func()
            
            if success:
                print("\n🔍 Verifying weights...")
                verify_weights()
            else:
                print("\n💡 Try a different method or download manually")
        else:
            print("❌ Invalid choice")
    except ValueError:
        print("❌ Invalid input")

if __name__ == "__main__":
    main()