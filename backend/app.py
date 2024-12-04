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