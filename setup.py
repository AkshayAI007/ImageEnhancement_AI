# PyTorch Compatibility Fix for Real-ESRGAN
# This fixes the 'torchvision.transforms.functional_tensor' error

Write-Host "🔧 Fixing PyTorch Compatibility Issues" -ForegroundColor Green
Write-Host "=" * 50

# Step 1: Complete cleanup of existing PyTorch installations
Write-Host "🧹 Step 1: Cleaning up existing PyTorch installations..." -ForegroundColor Yellow
pip uninstall -y torch torchvision torchaudio basicsr realesrgan gfpgan facexlib

# Step 2: Clear pip cache to avoid conflicts
Write-Host "🗑️ Step 2: Clearing pip cache..." -ForegroundColor Yellow
pip cache purge

# Step 3: Install compatible PyTorch versions
Write-Host "📦 Step 3: Installing compatible PyTorch versions..." -ForegroundColor Yellow

# Install specific compatible versions that work with Real-ESRGAN
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116

# Alternative for CPU-only (if GPU version fails)
# pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1

# Step 4: Install basicsr with specific version
Write-Host "🔧 Step 4: Installing basicsr..." -ForegroundColor Yellow
pip install basicsr==1.4.2

# Step 5: Install other dependencies
Write-Host "📦 Step 5: Installing other dependencies..." -ForegroundColor Yellow
pip install facexlib==0.2.5
pip install gfpgan==1.3.8
pip install opencv-python==4.7.1.72
pip install realesrgan==0.3.0

# Step 6: Install Flask dependencies
Write-Host "🌐 Step 6: Installing Flask dependencies..." -ForegroundColor Yellow
pip install Flask==2.3.3 flask-cors==4.0.0 Pillow==10.0.1 numpy==1.24.3

Write-Host "✅ Installation completed!" -ForegroundColor Green

# Step 7: Test the installation
Write-Host "🧪 Step 7: Testing installation..." -ForegroundColor Yellow

$testScript = @"
import sys
print(f'Python version: {sys.version}')

try:
    import torch
    print(f'✅ PyTorch version: {torch.__version__}')
    print(f'✅ CUDA available: {torch.cuda.is_available()}')
    
    import torchvision
    print(f'✅ Torchvision version: {torchvision.__version__}')
    
    # Test the problematic import
    try:
        import torchvision.transforms.functional_tensor
        print('✅ torchvision.transforms.functional_tensor imported successfully')
    except ImportError:
        print('⚠️  torchvision.transforms.functional_tensor not available, but this is expected in newer versions')
    
    from basicsr.archs.rrdbnet_arch import RRDBNet
    print('✅ basicsr imported successfully')
    
    from realesrgan import RealESRGANer
    print('✅ RealESRGANer imported successfully')
    
    from gfpgan import GFPGANer  
    print('✅ GFPGANer imported successfully')
    
    print('')
    print('🎉 All imports successful! Real-ESRGAN is ready to use.')
    
except ImportError as e:
    print(f'❌ Import error: {e}')
    print('')
    print('💡 Try the alternative installation method below.')
    
except Exception as e:
    print(f'❌ Unexpected error: {e}')
"@

$testScript | python

Write-Host ""
Write-Host "🎯 Next Steps:" -ForegroundColor Cyan
Write-Host "1. If the test above was successful, run: python api.py"
Write-Host "2. If there are still errors, try the alternative method below"