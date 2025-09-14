import tensorflow as tf
from tensorflow.keras import layers, Sequential

class Projector(tf.keras.Model):
    def __init__(self, input_dim, output_dim=128, name="Projector"):
        super(Projector, self).__init__(name=name)
        self.net = Sequential([
            layers.Dense(1024, activation='gelu'),    # Hidden layer
            layers.Dense(output_dim, activation=None)  # Output layer (128-dim)
        ])

    @tf.function(jit_compile=True)
    def call(self, x):
        return self.net(x)