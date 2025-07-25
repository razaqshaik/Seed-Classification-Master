from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Classes for classification
CLASSES = ['Broken', 'Immature', 'Intact', 'Skin-Damaged', 'Spotted']

# Load models
print("Loading models...")
# Load DenseNet model for feature extraction
densenet_model = DenseNet121(include_top=False, input_shape=(224, 224, 3), pooling='avg')
# Load your trained classifier
classifier = load_model('inception_v3.keras')
print("Models loaded successfully!")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def predict_seed_class(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    # Extract features using DenseNet
    features = densenet_model.predict(img_array)
    
    # Predict using the classifier
    predictions = classifier.predict(features)
    
    # Get class label and confidence
    class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][class_idx])
    
    return {
        'class': CLASSES[class_idx],
        'confidence': confidence,
        'all_probabilities': {CLASSES[i]: float(predictions[0][i]) for i in range(len(CLASSES))}
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            result = predict_seed_class(filepath)
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)})
    
    return jsonify({'error': 'File type not allowed'})

if __name__ == '__main__':
    app.run(debug=True)