import sys
import os

# CRITICAL: Apply PyTorch compatibility fix FIRST
def apply_pytorch_fix():
    """Apply PyTorch compatibility fix for functional_tensor module"""
    try:
        import torchvision.transforms.functional_tensor
        print("DEBUG: torchvision.transforms.functional_tensor already available")
    except ImportError:
        print("DEBUG: Applying PyTorch compatibility fix...")
        try:
            import torchvision.transforms.functional as F
            sys.modules['torchvision.transforms.functional_tensor'] = F
            print("DEBUG: ✅ PyTorch compatibility fix applied")
        except Exception as e:
            print(f"DEBUG: ❌ Could not apply fix: {e}")

# Apply fix before any other imports
apply_pytorch_fix()

try:
    from flask import Flask, request, send_file, jsonify, render_template
    from flask_cors import CORS
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps
    import numpy as np
    import io
    import gc
    import time
    import cv2
    print("DEBUG: Basic imports successful")
    
    # Try Real-ESRGAN imports
    REALESRGAN_AVAILABLE = False
    realesrgan_error = None
    
    try:
        import torch
        print(f"DEBUG: PyTorch {torch.__version__}")
        
        from basicsr.archs.rrdbnet_arch import RRDBNet
        print("DEBUG: basicsr imported")
        
        from realesrgan import RealESRGANer
        print("DEBUG: RealESRGANer imported")
        
        try:
            from gfpgan import GFPGANer
            print("DEBUG: GFPGANer imported")
            GFPGAN_AVAILABLE = True
        except:
            print("DEBUG: GFPGANer not available")
            GFPGAN_AVAILABLE = False
        
        REALESRGAN_AVAILABLE = True
        print("DEBUG: ✅ Real-ESRGAN components loaded successfully!")
        
    except Exception as e:
        REALESRGAN_AVAILABLE = False
        realesrgan_error = str(e)
        print(f"DEBUG: ❌ Real-ESRGAN not available: {e}")
        print("DEBUG: Will use PIL fallback")

except Exception as e:
    print(f"FATAL: Import error: {e}")
    exit(1)

# Configuration
MAX_IMAGE_SIZE = 1024
MAX_FILE_SIZE = 10 * 1024 * 1024
WEIGHTS_FOLDER = 'weights'
os.makedirs(WEIGHTS_FOLDER, exist_ok=True)

app = Flask(__name__)
CORS(app, origins="*")

# Global variables
realesrgan_upsampler = None
device = None

def download_weights():
    """Download model weights"""
    weights_path = os.path.join(WEIGHTS_FOLDER, "RealESRGAN_x4plus.pth")
    if not os.path.exists(weights_path):
        print("DEBUG: Downloading RealESRGAN weights...")
        try:
            import urllib.request
            url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
            urllib.request.urlretrieve(url, weights_path)
            print(f"DEBUG: ✅ Downloaded to {weights_path}")
            return True
        except Exception as e:
            print(f"DEBUG: ❌ Download failed: {e}")
            return False
    return True

# Initialize Real-ESRGAN
if REALESRGAN_AVAILABLE:
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"DEBUG: Using device: {device}")
        
        if download_weights():
            weights_path = os.path.join(WEIGHTS_FOLDER, "RealESRGAN_x4plus.pth")
            
            # Create model
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, 
                          num_block=23, num_grow_ch=32, scale=4)
            
            # Initialize upsampler
            realesrgan_upsampler = RealESRGANer(
                scale=4,
                model_path=weights_path,
                model=model,
                tile=256,
                tile_pad=10,
                pre_pad=0,
                half=False,  # Disable half precision for compatibility
                device=device
            )
            print("DEBUG: ✅ RealESRGAN initialized successfully!")
        else:
            REALESRGAN_AVAILABLE = False
            
    except Exception as e:
        print(f"DEBUG: ❌ RealESRGAN initialization failed: {e}")
        REALESRGAN_AVAILABLE = False
        realesrgan_error = str(e)
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps


def resize_if_needed(img, max_size):
    """Resize image if too large"""
    w, h = img.size
    if max(w, h) > max_size:
        if w > h:
            new_w, new_h = max_size, int(h * max_size / w)
        else:
            new_w, new_h = int(w * max_size / h), max_size
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        print(f"DEBUG: Resized to {new_w}x{new_h}")
    return img

def enhance_with_realesrgan(img):
    """Enhance image with Real-ESRGAN"""
    if not REALESRGAN_AVAILABLE or realesrgan_upsampler is None:
        raise ValueError("Real-ESRGAN not available")
    
    # Convert to BGR for OpenCV
    img_array = np.array(img)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Enhance with Real-ESRGAN
    with torch.no_grad():
        enhanced_bgr, _ = realesrgan_upsampler.enhance(img_bgr, outscale=4)
    
    # Convert back to RGB
    enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(enhanced_rgb)

def pil_fallback_enhance(img):
    """PIL fallback enhancement"""
    print("DEBUG: Using PIL fallback...")
    
    # Preprocess
    img = advanced_preprocessing(img)
    
    # 4x upscale
    w, h = img.size
    img = img.resize((w * 4, h * 4), Image.Resampling.LANCZOS)
    
    # Additional enhancement
    img = ImageEnhance.Sharpness(img).enhance(1.3)
    img = ImageEnhance.Contrast(img).enhance(1.2)
    img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    
    return img

@app.route('/api/enhance', methods=['POST'])
def enhance_image():
    print("DEBUG: Enhancement request received")
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    # Check file size
    file.seek(0, os.SEEK_END)
    if file.tell() > MAX_FILE_SIZE:
        return jsonify({'error': 'File too large'}), 400
    file.seek(0)
    
    try:
        # Load image
        img = Image.open(file.stream).convert('RGB')
        print(f"DEBUG: Image loaded: {img.size}")
        
        # Resize if needed
        img = resize_if_needed(img, MAX_IMAGE_SIZE)
        
        # Enhance image
        start_time = time.time()
        
        if REALESRGAN_AVAILABLE:
            print("DEBUG: Using Real-ESRGAN...")
            # Preprocess first
            preprocessed = img
            # Then enhance with AI
            enhanced = enhance_with_realesrgan(preprocessed)
            # Light post-processing
            enhanced = ImageEnhance.Sharpness(enhanced).enhance(1.02)
            method = "Real-ESRGAN 4x + Preprocessing"
        else:
            print("DEBUG: Using PIL fallback...")
            enhanced = pil_fallback_enhance(img)
            method = f"PIL Enhancement (Real-ESRGAN unavailable: {realesrgan_error})"
        
        processing_time = time.time() - start_time
        print(f"DEBUG: Enhancement completed in {processing_time:.2f}s")
        
        # Save result
        byte_io = io.BytesIO()
        if enhanced.size[0] * enhanced.size[1] > 8000000:
            enhanced.save(byte_io, 'JPEG', quality=95, optimize=True)
            mimetype = 'image/jpeg'
        else:
            enhanced.save(byte_io, 'PNG', optimize=True)
            mimetype = 'image/png'
        
        byte_io.seek(0)
        
        response = send_file(byte_io, mimetype=mimetype)
        response.headers['X-Processing-Method'] = method
        response.headers['X-Processing-Time'] = f"{processing_time:.2f}s"
        
        # Cleanup
        del img, enhanced
        gc.collect()
        if device and device.type == 'cuda':
            torch.cuda.empty_cache()
        
        return response
        
    except Exception as e:
        print(f"DEBUG: Enhancement failed: {e}")
        return jsonify({'error': f'Enhancement failed: {str(e)}'}), 500

@app.route('/api/status')
def get_status():
    return jsonify({
        'status': 'healthy',
        'realesrgan_available': REALESRGAN_AVAILABLE,
        'realesrgan_error': realesrgan_error if not REALESRGAN_AVAILABLE else None,
        'device': str(device) if device else 'N/A',
        'cuda_available': torch.cuda.is_available() if REALESRGAN_AVAILABLE else False,
        'enhancement_method': 'Real-ESRGAN 4x + Preprocessing' if REALESRGAN_AVAILABLE else 'PIL Advanced Enhancement'
    })

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    print(f"DEBUG: Starting API (Real-ESRGAN: {'✅' if REALESRGAN_AVAILABLE else '❌'})")
    if REALESRGAN_AVAILABLE:
        print(f"DEBUG: Device: {device}")
    else:
        print(f"DEBUG: Error: {realesrgan_error}")
    
    app.run(host="0.0.0.0", port=5000, debug=True)