from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

def build_mlp_model():
    """Construye un modelo de red neuronal multicapa (MLP)"""
    model = Sequential([
        Flatten(input_shape=(784, )),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    return model