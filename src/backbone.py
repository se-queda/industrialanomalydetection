import tensorflow as tf


def backbone_model(input_shape=(256, 256, 3)):
    base_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
    base_model.trainable = False
    layer_names = ["conv2_block3_out", "conv3_block4_out"]
    outputs = [base_model.get_layer(name).output for name in layer_names]
    model = tf.keras.Model(inputs=base_model.input, outputs=outputs, name="ResNet50_Backbone")

    return model
