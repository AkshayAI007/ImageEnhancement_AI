# test_imports.py - Simple test file
import sys

print("Testing Real-ESRGAN imports...")
print("=" * 40)

try:
    # Apply compatibility fix first
    try:
        import torchvision.transforms.functional_tensor
        print("✅ torchvision.transforms.functional_tensor available")
    except ImportError:
        import torchvision.transforms.functional as F
        sys.modules['torchvision.transforms.functional_tensor'] = F
        print("✅ Applied compatibility fix for functional_tensor")
    
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    
    import torchvision
    print(f"✅ Torchvision: {torchvision.__version__}")
    
    from basicsr.archs.rrdbnet_arch import RRDBNet
    print("✅ basicsr imported")
    
    from realesrgan import RealESRGANer
    print("✅ RealESRGANer imported")
    
    from gfpgan import GFPGANer
    print("✅ GFPGANer imported")
    
    print("\n🎉 All imports successful!")
    print("✅ Ready to use Real-ESRGAN!")
    
except Exception as e:
    print(f"❌ Import failed: {e}")
    print("\n💡 Try installing older versions:")
    print("pip install torch==1.11.0 torchvision==0.12.0")