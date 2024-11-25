import numpy as np
from tensorflow.keras.utils import to_categorical

def preprocess_data_for_mlp(x_train, x_test):
    """Realiza el preprocesamiento para la MLP (aplanado de imágenes)."""
    # Aplanar las imágenes de 28x28 a 784 (28*28) y normalizar los valores (0 a 255) a un rango de 0 a 1
    x_train_mlp = x_train.reshape(x_train.shape[0], -1).astype('float32') / 255.0
    x_test_mlp = x_test.reshape(x_test.shape[0], -1).astype('float32') / 255.0

    return x_train_mlp, x_test_mlp

def preprocess_data_for_cnn(x_train, x_test):
    """Realiza el preprocesamiento para la CNN (mantener imágenes 2D y normalizar)."""
    # Normalizar las imágenes (su valor de cada pixel va de 0 a 255) entre 0 y 1
    x_train_cnn = x_train.astype('float32') / 255.0
    x_test_cnn = x_test.astype('float32') / 255.0

    # Adicion de una dimension a cada imagen estén en el formato adecuado para las CNN (28x28x1)
    # (28, 28) para las imagenes
    # 1 para el canal de color (escala de grises)
    x_train_cnn = np.expand_dims(x_train_cnn, axis=-1)
    x_test_cnn = np.expand_dims(x_test_cnn, axis=-1)

    return x_train_cnn, x_test_cnn

def preprocess_labels(y_train, y_test):
    """Realiza el preprocesamiento de las etiquetas (one-hot encoding)."""
    # Convertir las etiquetas a one-hot encoding
    y_train_one_hot = to_categorical(y_train, num_classes=10)
    y_test_one_hot = to_categorical(y_test, num_classes=10)

    return y_train_one_hot, y_test_one_hot