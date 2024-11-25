import tensorflow as tf
import numpy as np
import os
from src.data_preprocessing import *

def download_raw_data():
    """Descarga MNIST original y lo guarda en /data/raw en formato raw"""
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Directorio para guardar MNIST y comprobacion de su existencia
    output_dir = os.path.join('.', 'data', 'raw')
    os.makedirs(output_dir, exist_ok=True)

    # Rutas de guardado para las imagenes y etiquetas de entrenamiento y testeo 
    train_images_file = os.path.join(output_dir, 'train_images.raw')
    train_labels_file = os.path.join(output_dir, 'train_labels.raw')
    test_images_file = os.path.join(output_dir, 'test_images.raw')
    test_labels_file = os.path.join(output_dir, 'test_labels.raw')

    # Guardado de las imagenes y etiquetas en formato raw
    x_train.tofile(train_images_file)
    y_train.tofile(train_labels_file)
    x_test.tofile(test_images_file)
    y_test.tofile(test_labels_file)

    print("Imagenes y etiquetas de MNIST guardados (formato raw) en /data/raw")

def load_data():
    """Carga los datos de MNIST en formato raw"""
    raw_dir = os.path.join('.', 'data', 'raw')

    # Rutas de los archivos raw 
    train_images_file = os.path.join(raw_dir, 'train_images.raw')
    train_labels_file = os.path.join(raw_dir, 'train_labels.raw')
    test_images_file = os.path.join(raw_dir, 'test_images.raw')
    test_labels_file = os.path.join(raw_dir, 'test_labels.raw')

    # Cargar imagenes y etiquetas desde los archivos raw 
    train_images = np.fromfile(train_images_file, dtype=np.uint8).reshape(-1, 28, 28)
    train_labels = np.fromfile(train_labels_file, dtype=np.uint8)
    test_images = np.fromfile(test_images_file, dtype=np.uint8).reshape(-1, 28, 28)
    test_labels = np.fromfile(test_labels_file, dtype=np.uint8)

    print("Datos MNIST cargados (y procesados) desde /data/raw")
    return (train_images, train_labels), (test_images, test_labels)

def save_preprocessed_data(x_train_mlp, x_test_mlp, x_train_cnn, x_test_cnn, y_train, y_test):
    """Guarda los datos preprocesados en /data/preprocessed"""
    preprocessed_dir = os.path.join('.', 'data', 'preprocessed')
    os.makedirs(preprocessed_dir, exist_ok=True)

    # Rutas de guardado para las imagenes preprocesadas
    train_mlp_file = os.path.join(preprocessed_dir, 'train_mlp.npy')
    test_mlp_file = os.path.join(preprocessed_dir, 'test_mlp.npy')
    train_cnn_file = os.path.join(preprocessed_dir, 'train_cnn.npy')
    test_cnn_file = os.path.join(preprocessed_dir, 'test_cnn.npy')

    # Rutas de guardado para las etiquetas preprocesadas
    train_labels_file = os.path.join(preprocessed_dir, 'train_labels.npy')
    test_labels_file = os.path.join(preprocessed_dir, 'test_labels.npy')

    # Guardar las imagenes y etiquetas preprocesadas
    np.save(train_mlp_file, x_train_mlp)
    np.save(test_mlp_file, x_test_mlp)
    np.save(train_cnn_file, x_train_cnn)
    np.save(test_cnn_file, x_test_cnn)
    np.save(train_labels_file, y_train)
    np.save(test_labels_file, y_test)

def run_data_pipeline():
    """Organiza la descarga, carga y procesamiento de los datos sde MNIST"""
    # Descargar y guardar los datos de MNIST en formato raw
    download_raw_data()

    # Cargar los datos raw de MNIST
    (x_train, y_train), (x_test, y_test) = load_data()

    # Preprocesar los datos para una MLP
    x_train_mlp, x_test_mlp = preprocess_data_for_mlp(x_train, x_test)
    print("Datos preprocesados para MLP listos.")

    # Preprocesar los datos para una CNN
    x_train_cnn, x_test_cnn = preprocess_data_for_cnn(x_train, x_test)
    print("Datos preprocesados para CNN listos.")

    # Preprocesar las etiquetas (formato one-hot encoding)
    y_train_one_hot, y_test_one_hot = preprocess_labels(y_train, y_test)
    print("Etiquetas preprocesadas listas.")

    # Guardar los datos preprocesados
    save_preprocessed_data(x_train_mlp, x_test_mlp, x_train_cnn, x_test_cnn, y_train_one_hot, y_test_one_hot)
    print("Imagenes y etiquetas preprocesadas guardadas en /data/preprocessed")

