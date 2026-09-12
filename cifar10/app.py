import os
import sys
import time
import logging
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

# Suppress TensorFlow informational logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import load_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask application
app = Flask(__name__)

# Security & Upload configurations
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# Base project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# CIFAR-10 10 Classes in the exact order the model was trained on
CLASS_NAMES = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]

# Emoji mappings for enhanced UI visualization
CLASS_EMOJIS = {
    'airplane': '✈️',
    'automobile': '🚗',
    'bird': '🐦',
    'cat': '🐱',
    'deer': '🦌',
    'dog': '🐶',
    'frog': '🐸',
    'horse': '🐴',
    'ship': '🚢',
    'truck': '🚚'
}

# Category descriptions to help users understand the model's domain
CLASS_DESCRIPTIONS = {
    'airplane': 'Commercial aircraft, jet planes, airplanes',
    'automobile': 'Sedans, passenger cars, sports cars',
    'bird': 'Birds of all species, perching or flying',
    'cat': 'Domestic cats and felines',
    'deer': 'Stags, does, wild deer',
    'dog': 'Dogs and domestic canines of various breeds',
    'frog': 'Frogs, toads, amphibians',
    'horse': 'Horses, stallions, ponies, equines',
    'ship': 'Boats, cargo vessels, watercraft',
    'truck': 'Pickup trucks, freight trucks, lorries'
}

def allowed_file(filename):
    """Validate that the uploaded file has a permissible extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_model_path():
    """Locate the trained model file in the workspace."""
    candidate_paths = [
        os.path.join(BASE_DIR, 'Cifar10.h5'),
        os.path.join(BASE_DIR, 'cifar10_model.keras'),
        os.path.join(BASE_DIR, 'cifar10_model.h5'),
    ]
    for path in candidate_paths:
        if os.path.isfile(path):
            return path
    return None

# Load model once at startup
MODEL_PATH = get_model_path()
model = None

if MODEL_PATH:
    try:
        logger.info(f"Loading trained model from {MODEL_PATH}...")
        model = load_model(MODEL_PATH)
        logger.info("Model loaded successfully.")
        
        # Model warm-up: Run a dummy inference pass so first user request is instant
        dummy_tensor = np.zeros((1, 32, 32, 3), dtype=np.float32)
        _ = model.predict(dummy_tensor, verbose=0)
        logger.info("Model warm-up completed successfully.")
    except Exception as e:
        logger.error(f"Failed to load model from {MODEL_PATH}: {e}")
        model = None
else:
    logger.warning("No trained model file found (Cifar10.h5 or cifar10_model.keras).")

def preprocess_image(image_source):
    """
    Preprocess image exactly matching the CIFAR-10 training pipeline:
    1. Open image using PIL
    2. Convert to RGB (handles RGBA, grayscale, CMYK)
    3. Resize to exactly 32x32 pixels
    4. Convert to NumPy float32 array
    5. Normalize pixel values to [0.0, 1.0] by dividing by 255.0
    6. Expand dimensions to (1, 32, 32, 3)
    """
    image = Image.open(image_source)
    image_rgb = image.convert('RGB')
    image_resized = image_rgb.resize((32, 32), Image.Resampling.BILINEAR)
    
    image_array = np.array(image_resized, dtype=np.float32) / 255.0
    image_batch = np.expand_dims(image_array, axis=0)
    
    return image_batch, image.size

def run_inference(image_batch):
    """
    Run model inference and extract top predictions.
    """
    if model is None:
        raise RuntimeError("Model is not initialized or failed to load.")
    
    start_time = time.perf_counter()
    raw_predictions = model.predict(image_batch, verbose=0)[0]
    inference_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
    
    # Calculate percentage probabilities
    probabilities = raw_predictions * 100.0
    
    # Sort indices by probability descending
    sorted_indices = np.argsort(probabilities)[::-1]
    
    top_idx = int(sorted_indices[0])
    primary_class = CLASS_NAMES[top_idx]
    primary_confidence = float(round(probabilities[top_idx], 2))
    
    # Top 3 predictions
    top_predictions = []
    for rank, idx in enumerate(sorted_indices[:3], start=1):
        c_name = CLASS_NAMES[int(idx)]
        top_predictions.append({
            'rank': rank,
            'class': c_name,
            'displayName': c_name.capitalize(),
            'emoji': CLASS_EMOJIS.get(c_name, '🔍'),
            'confidence': float(round(probabilities[int(idx)], 2))
        })
    
    return {
        'prediction': primary_class,
        'displayName': primary_class.capitalize(),
        'emoji': CLASS_EMOJIS.get(primary_class, '🔍'),
        'confidence': primary_confidence,
        'top_predictions': top_predictions,
        'inference_time_ms': inference_time_ms
    }

@app.route('/', methods=['GET'])
def index():
    """Render main application dashboard."""
    classes_info = [
        {
            'name': name,
            'displayName': name.capitalize(),
            'emoji': CLASS_EMOJIS.get(name, '🔍'),
            'description': CLASS_DESCRIPTIONS.get(name, '')
        }
        for name in CLASS_NAMES
    ]
    return render_template('index.html', classes=classes_info, model_loaded=(model is not None))

@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint.
    Accepts multipart/form-data with 'image' or 'file' key.
    Returns JSON prediction results or friendly error details.
    """
    if model is None:
        return jsonify({
            'success': False,
            'error': 'The CIFAR-10 model is not currently loaded on the server. Please verify the model file exists.'
        }), 500

    # Retrieve file from request
    file = request.files.get('image') or request.files.get('file')
    if not file or file.filename == '':
        return jsonify({
            'success': False,
            'error': 'Please select or drop an image file first.'
        }), 400

    filename = secure_filename(file.filename)
    if not allowed_file(filename):
        return jsonify({
            'success': False,
            'error': 'Unsupported image format. Please upload a JPG, JPEG, PNG, or WEBP image.'
        }), 400

    try:
        # Preprocess uploaded image
        image_batch, original_dimensions = preprocess_image(file.stream)
        
        # Perform inference
        result = run_inference(image_batch)
        
        return jsonify({
            'success': True,
            'prediction': result['prediction'],
            'displayName': result['displayName'],
            'emoji': result['emoji'],
            'confidence': result['confidence'],
            'top_predictions': result['top_predictions'],
            'inference_time_ms': result['inference_time_ms'],
            'original_dimensions': {
                'width': original_dimensions[0],
                'height': original_dimensions[1]
            },
            'model_input_shape': '32×32×3'
        }), 200

    except Image.UnidentifiedImageError:
        logger.warning(f"Failed to identify/decode image file: {filename}")
        return jsonify({
            'success': False,
            'error': 'We could not decode this image file. It may be damaged or in an unsupported format.'
        }), 400
    except Exception as e:
        logger.error(f"Inference error processing {filename}: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Something went wrong while analyzing the image. Please try again.'
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check route."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_path': MODEL_PATH,
        'framework': 'TensorFlow / Keras',
        'cifar10_classes_count': len(CLASS_NAMES)
    }), 200

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        'success': False,
        'error': 'The uploaded file is too large. Maximum supported size is 10 MB.'
    }), 413

@app.errorhandler(404)
def not_found_error(error):
    if request.path.startswith('/predict') or request.path.startswith('/health'):
        return jsonify({'success': False, 'error': 'Endpoint not found.'}), 404
    return "Page not found", 404

@app.errorhandler(500)
def internal_server_error(error):
    return jsonify({
        'success': False,
        'error': 'An internal server error occurred. Please check server logs.'
    }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting CIFAR Vision Flask app on http://127.0.0.1:{port} ...")
    app.run(host='0.0.0.0', port=port, debug=False)
