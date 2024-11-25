import numpy as np

def preprocess_data_for_mlp(x_train, x_test):
    """Realiza el preprocesamiento para la MLP (aplanado de imágenes)."""
    # Aplanar las imágenes de 28x28 a 784 (28*28) y normalizar los valores (0 a 255) a un rango de 0 a 1
    x_train_mlp = x_train.reshape(x_train.shape[0], -1).astype('float32') / 255.0
    x_test_mlp = x_test.reshape(x_test.shape[0], -1).astype('float32') / 255.0

    return x_train_mlp, x_test_mlp