import tensorflow as tf


def extract_patches(features, patch_size=3, stride=1):
    """
    XLA-compatible patch extraction.
    """
    B, H, W, C = tf.shape(features)[0], tf.shape(features)[1], tf.shape(features)[2], tf.shape(features)[3]

    # Create grid of patch starting points
    patch_H = (H - patch_size) // stride + 1
    patch_W = (W - patch_size) // stride + 1

    # Generate indices for the top-left corner of each patch
    y, x = tf.meshgrid(tf.range(patch_H), tf.range(patch_W), indexing='ij')
    start_indices = tf.stack([y * stride, x * stride], axis=-1)
    start_indices = tf.reshape(start_indices, [patch_H * patch_W, 2])

    # Create offsets for a single patch
    patch_y, patch_x = tf.meshgrid(tf.range(patch_size), tf.range(patch_size), indexing='ij')
    patch_offsets = tf.stack([patch_y, patch_x], axis=-1)
    patch_offsets = tf.reshape(patch_offsets, [1, patch_size * patch_size, 2])

    # Combine start indices and offsets to get absolute indices for all patches
    all_indices = tf.expand_dims(start_indices, 1) + patch_offsets

    # Add batch dimension
    batch_indices = tf.expand_dims(tf.range(B), 1)
    batch_indices = tf.tile(batch_indices, [1, patch_H * patch_W * patch_size * patch_size])
    all_indices = tf.tile(all_indices, [B, 1, 1])
    all_indices = tf.reshape(all_indices, (B, -1, 2))

    # Gather patches
    patches = tf.gather_nd(features, all_indices, batch_dims=1)

    # Reshape to final output
    patches = tf.reshape(patches, [B, patch_H * patch_W, patch_size * patch_size * C])

    return patches


def aggregate(feat_block2, feat_block3):
    patches_b2 = extract_patches(feat_block2, patch_size=3, stride=1)
    patches_b3 = extract_patches(feat_block3, patch_size=3, stride=1)
    return patches_b2, patches_b3