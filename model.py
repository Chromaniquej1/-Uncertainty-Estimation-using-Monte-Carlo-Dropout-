import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Dropout
from tensorflow.keras.initializers import lecun_normal

def build_nn_model():
    model = Sequential([InputLayer(input_shape=(1,)),
                        Dropout(0.4),
                        Dense(1000, activation="selu", kernel_initializer="lecun_normal"),
                        Dropout(0.4),
                        Dense(2000, activation="selu", kernel_initializer="lecun_normal"),
                        Dropout(0.4),
                        Dense(2000, activation="selu", kernel_initializer="lecun_normal"),
                        Dropout(0.4),
                        Dense(1000, activation="selu", kernel_initializer="lecun_normal"),
                        Dropout(0.4),
                        Dense(1, activation=None)
                        ])
    return model

class MCDropout(Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True)

def build_mc_model():
    model = Sequential([InputLayer(input_shape=(1,)),
                        MCDropout(0.4),
                        Dense(1000, activation="selu", kernel_initializer="lecun_normal"),
                        MCDropout(0.4),
                        Dense(2000, activation="selu", kernel_initializer="lecun_normal"),
                        MCDropout(0.4),
                        Dense(2000, activation="selu", kernel_initializer="lecun_normal"),
                        MCDropout(0.4),
                        Dense(1000, activation="selu", kernel_initializer="lecun_normal"),
                        MCDropout(0.4),
                        Dense(1, activation=None)
                        ])
    return model
