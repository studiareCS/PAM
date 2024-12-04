from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import os

app = Flask(__name__)

# Ruta a los modelos entrenados
MODEL_DIR = os.path.join(os.path.dirname(__file__), '../models')
cnn_model_path = os.path.join(MODEL_DIR, 'cnn_model.h5')
mlp_model_path = os.path.join(MODEL_DIR, 'mlp_model.h5')

# Cargar modelos
cnn_model = tf.keras.models.load_model(cnn_model_path)
mlp_model = tf.keras.models.load_model(mlp_model_path)

@app.route('/')
def index():
    """Renderiza la página principal."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Recibe una imagen dibujada por el usuario y realiza predicciones."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400