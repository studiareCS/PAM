import tensorflow as tf
import numpy as np
import os

def download_raw_data():
    """Descarga MNIST original y lo guarda en /data/raw en formato raw"""
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Directorio para guardar MNIST y comprobacion de su existencia
    output_dir = './data/raw'
    os.makedirs(output_dir, exist_ok=True)