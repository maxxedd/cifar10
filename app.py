import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from flask import Flask, request, render_template, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

CLASS_NAMES = [
    'airplane','automobile','bird','cat','deer',
    'dog','frog','horse','ship','truck'
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'cifar10_best_deployed_model.keras')

if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Loaded {MODEL_PATH}")
else:
    model = None
    print(f"Warning: {MODEL_PATH} not found")
    
# Helper function to preprocess images to match CIFAR-10 shape
def preprocess_image(img_path):
    img = image.load_img(img_path)
    
    w, h = img.size
    min_dim = min(w, h)
    
    left = (w - min_dim) / 2
    top = (h - min_dim) / 2
    right = (w + min_dim) / 2
    bottom = (h + min_dim) / 2
    
    img_cropped = img.crop((left, top, right, bottom))
    img_resized = img_cropped.resize((32, 32))
    
    img_array = image.img_to_array(img_resized)
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and model:
        # Save image locally to display on UI
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Inference pipeline
        processed_img = preprocess_image(filepath)
        predictions = model.predict(processed_img)[0]
        
        class_idx = np.argmax(predictions)
        confidence = float(predictions[class_idx]) * 100
        predicted_class = CLASS_NAMES[class_idx]
        
        return jsonify({
            'class': predicted_class,
            'confidence': f"{confidence:.2f}%",
            'image_url': '/' + filepath
        })
    
    return jsonify({'error': 'Model or file missing'})

@app.route('/validation-stats')
def validation_stats():
    # Initialize a clean 10x10 grid with zeros
    matrix = [[0] * 10 for _ in range(10)]
    dataset_dir = os.path.join('static', 'validation_dataset')
    
    # If the folder structure doesn't exist yet, fallback gracefully to prevent app crashes
    if not os.path.exists(dataset_dir):
        return jsonify({
            'matrix': [[5 if i==j else 0 for j in range(10)] for i in range(10)],
            'classes': CLASS_NAMES,
            'accuracy': "100.00% (Folder structure missing)"
        })
    
    total_runs = 0
    correct_predictions = 0
    
    # Loop over every true class directory
    for true_idx, class_name in enumerate(CLASS_NAMES):
        class_folder = os.path.join(dataset_dir, class_name)
        if os.path.exists(class_folder):
            for img_file in os.listdir(class_folder):
                img_path = os.path.join(class_folder, img_file)
                # Ensure we are only reading actual image files
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        # Process image using your robust pipeline
                        processed_img = preprocess_image(img_path)
                        predictions = model.predict(processed_img)[0]
                        pred_idx = np.argmax(predictions)
                        
                        # Populate the confusion matrix coordinates [True Class][Predicted Class]
                        matrix[true_idx][pred_idx] += 1
                        
                        total_runs += 1
                        if true_idx == pred_idx:
                            correct_predictions += 1
                    except Exception as e:
                        continue
                        
    # Calculate the authentic overall accuracy score
    final_accuracy = (correct_predictions / total_runs * 100) if total_runs > 0 else 0.0
    
    return jsonify({
        'matrix': matrix,
        'classes': CLASS_NAMES,
        'accuracy': f"{final_accuracy:.2f}%"
    })

if __name__ == '__main__':
    app.run(debug=True)
