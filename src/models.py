from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import yaml
import os

# Abrir el archivo de configuracion
with open(os.path.join('.', 'config', 'config.yaml'), 'r') as file:
    config = yaml.safe_load(file)

# Variables globales
relu = config['activation_functions'][0]['relu']
softmax = config['activation_functions'][1]['softmax']

def build_mlp_model():
    """Construye un modelo de red neuronal multicapa (MLP)"""
    mlp_model = Sequential([
        Flatten(input_shape=(784, )),
        Dense(256, activation=relu),
        Dense(128, activation=relu),
        Dense(64, activation=relu),
        Dense(10, activation=softmax)
    ])
    return mlp_model

def build_cnn_model():
    """Construye un modelo de red neuronal convolucional (CNN)"""
    cnn_model = Sequential([
        Conv2D(32, (3, 3), activation=relu, input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation=relu),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation=relu),
        Dense(10, activation=softmax)
    ])
    return cnn_model

# Cerrar el archivo de configuracion
file.close()