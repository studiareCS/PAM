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

def run_data_pipeline():
    """Organiza la descarga, carga y procesamiento de los datos de MNIST"""
    # Descargar y guardar los datos de MNIST en formato raw
    download_raw_data()
