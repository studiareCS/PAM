import tensorflow as tf
import numpy as np
import os

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

    train_images_file = os.path.join(raw_dir, 'train_images.raw')
    train_labels_file = os.path.join(raw_dir, 'train_labels.raw')
    test_images_file = os.path.join(raw_dir, 'test_images.raw')
    test_labels_file = os.path.join(raw_dir, 'test_labels.raw')