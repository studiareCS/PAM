import os
import numpy as np
from tensorflow.keras.optimizers import Adam
from src.models import build_mlp_model, build_cnn_model
import yaml

# Abrir el archivo de configuracion
with open(os.path.join('.', 'config', 'config.yaml'), 'r') as file:
    config = yaml.safe_load(file)

# Variables globales
EPOCHS = config['training']['epochs']
BATCH_SIZE = config['training']['batch_size']
LEARNING_RATE = config['training']['learning_rate']
OPTIMIZER = config['training']['optimizer']
LOSS = config['training']['loss']
METRICS = config['training']['metrics']

def load_processed_data(model_type):
    """Carga los datos procesados de MNIST"""
    processed_dir = os.path.join('.', 'data', 'processed')

    # Cargar las imagenes preprocesados de MNIST segun el tipo de modelo
    if(model_type == 'mlp'):
        x_train = np.load(os.path.join(processed_dir, 'train_mlp.npy'))
        x_test = np.load(os.path.join(processed_dir, 'test_mlp.npy'))
    elif(model_type == 'cnn'):
        x_train = np.load(os.path.join(processed_dir, 'train_cnn.npy'))
        x_test = np.load(os.path.join(processed_dir, 'test_cnn.npy'))

    # Cargar las etiquetas preprocesadas de MNIST
    y_train = np.load(os.path.join(processed_dir, 'train_labels.npy'))
    y_test = np.load(os.path.join(processed_dir, 'test_labels.npy'))

    return x_train, x_test, y_train, y_test

def train_mlp():
    """Entrena un modelo de red neuronal multicapa (MLP) para MNIST y lo guarda"""
    # Cargar los datos preprocesados para la MLP
    x_train, x_test, y_train, y_test = load_processed_data('mlp')

    # Crear y compilar el modelo MLP
    mlp_model = build_mlp_model()
    mlp_model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE), 
        loss=LOSS, 
        metrics=METRICS
    )

    # Entrenar el modelo MLP
    mlp_model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(x_test, y_test))

    # Guardar el modelo entrenado
    os.makedirs(os.path.join('.', 'models'), exist_ok=True)
    mlp_model.save(os.path.join('.', 'models', 'mlp_model.h5'))
    print("Modelo MLP entrenado y guardado.")

def train_cnn():
    """Entrena un modelo de red neuronal convolucional (CNN) para MNIST y lo guarda"""
    # Cargar los datos preprocesados para la CNN
    x_train, x_test, y_train, y_test = load_processed_data('cnn')

    # Crear y compilar el modelo CNN
    cnn_model = build_cnn_model()
    cnn_model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE), 
        loss=LOSS, 
        metrics=METRICS
    )

    # Entrenar el modelo CNN
    cnn_model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(x_test, y_test))

    # Guardar el modelo entrenado
    os.makedirs(os.path.join('.', 'models'), exist_ok=True)
    cnn_model.save(os.path.join('.', 'models', 'cnn_model.h5'))
    print("Modelo CNN entrenado y guardado.")

if __name__ == '__main__':
    # Con los datos preprocesados de /data/preprocessed:
    # Se entrena y guarda el modelo MLP
    train_mlp()
    # Se entrena y guarda el modelo CNN
    train_cnn()

# Cerrar el archivo de configuracion
file.close()