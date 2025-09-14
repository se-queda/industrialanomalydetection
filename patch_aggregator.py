import tensorflow as tf


@tf.function(jit_compile=True)
def extract_patches(features, patch_size=3, stride=1):
    B = tf.shape(features)[0]
    C = tf.shape(features)[-1]

    patches = tf.image.extract_patches(
        images=features,
        sizes=[1, patch_size, patch_size, 1],
        strides=[1, stride, stride, 1],
        rates=[1, 1, 1, 1],
        padding='VALID'
    )

    patches = tf.reshape(patches, [B, -1, patch_size * patch_size * C])
    return patches
@tf.function(jit_compile=True)
def aggregate(feat_block2, feat_block3):
    patches_b2 = extract_patches(feat_block2, patch_size=3, stride=1)
    patches_b3 = extract_patches(feat_block3, patch_size=3, stride=1)
    return patches_b2, patches_b3