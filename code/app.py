import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, render_template, jsonify
import cv2
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

CLASSES = ["Glioma Tumor", "Meningioma Tumor", "Pituitary Tumor"]

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the models again to load state_dicts
def get_mobilenet_model():
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Sequential(
        nn.Linear(model.last_channel, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 3)
    )
    return model

# Load model
model_path = "brain_tumor_mobilenet.pth"
model = None
if os.path.exists(model_path):
    try:
        model = get_mobilenet_model()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print(f"Model loaded: {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")

# Preprocessing transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def preprocess_image(img_path):
    # Apply Histogram Equalization and Noise Reduction as in training
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
    processed_img = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
    
    # Convert to PIL Image for torchvision transforms
    pil_img = Image.fromarray(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
    input_tensor = transform(pil_img).unsqueeze(0).to(device)
    return input_tensor

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if file:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        
        if model is None:
            import random
            res_class = random.choice(CLASSES)
            res_conf = random.randint(80, 99)
            return jsonify({
                'class': res_class + " (Mock)",
                'confidence': res_conf
            })
        
        # Inference
        input_tensor = preprocess_image(filepath)
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, class_idx = torch.max(probabilities, 0)
            
        return jsonify({
            'class': CLASSES[class_idx.item()],
            'confidence': round(confidence.item() * 100, 2)
        })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
