import os
import numpy as np
from tensorflow.keras.optimizers import Adam
from src.models import build_mlp_model, build_cnn_model

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