import tensorflow as tf
from tensorflow.keras import layers, Sequential

class Projector(tf.keras.Model):
    def __init__(self, input_dim, output_dim=128, name="Projector"):
        super(Projector, self).__init__(name=name)
        # Add dtype='mixed_float16' to the layers
        self.net = Sequential([
            layers.Dense(1024, activation='gelu', dtype='mixed_float16'),
            layers.Dense(output_dim, activation=None, dtype='mixed_float16')
        ])

    @tf.function(jit_compile=True)
    def call(self, x):
        return self.net(x)