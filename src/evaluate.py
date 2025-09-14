import tensorflow as tf
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter
from src.patch_aggregator import aggregate

# =========================
# ⚡ Forward Pass (Evaluation)
# =========================
@tf.function
def forward_pass_eval(batch_images, backbone, f_proj, g_proj):
    """Backbone + projector forward pass for evaluation."""
    feat_block2, feat_block3 = backbone(batch_images, training=False)
    patches_b2, patches_b3 = aggregate(feat_block2, feat_block3)
    z_f = f_proj(patches_b2)
    z_g = g_proj(patches_b3)
    # Also return the shape of the feature maps for reshaping later
    return tf.concat([z_f, z_g], axis=1), (tf.shape(feat_block2)[1], tf.shape(feat_block2)[2])

# =========================
# 🧐 Evaluation Step
# =========================
@tf.function
def evaluate_step(batch_images, backbone, f_proj, g_proj, student_model):
    """Compiled evaluation step for faster inference."""
    (embeddings, patch_shapes) = forward_pass_eval(batch_images, backbone, f_proj, g_proj)
    student_embeddings = student_model(embeddings, training=False)
    return student_embeddings, patch_shapes


def evaluate(test_ds, backbone, f_proj, g_proj, student_model, memory_bank):
    """Calculate image-level and pixel-level AUROC on the test set."""
    y_true_img, y_scores_img = [], []
    y_true_pixel, y_scores_pixel = [], []

    for batch_images, labels, masks in tqdm(test_ds, desc="Running Test Validation"):
        student_embeddings, patch_shapes = evaluate_step(batch_images, backbone, f_proj, g_proj, student_model)

        # Calculate patch-level anomaly scores
        dists = memory_bank.query(student_embeddings.numpy().reshape(-1, 128), k=1)

        # Reshape to create an anomaly map for each image in the batch
        patch_scores = tf.reshape(dists, (tf.shape(batch_images)[0], patch_shapes[0], patch_shapes[1]))

        # Upsample anomaly map to image size
        anomaly_maps = tf.image.resize(patch_scores[:, :, :, tf.newaxis], [256, 256]).numpy()

        # Apply Gaussian smoothing
        for i in range(anomaly_maps.shape[0]):
            anomaly_maps[i] = gaussian_filter(anomaly_maps[i], sigma=4)

        # --- Image-level AUROC ---
        y_true_img.extend(labels.numpy())
        y_scores_img.extend(np.max(anomaly_maps, axis=(1, 2, 3)))

        # --- Pixel-level AUROC ---
        y_true_pixel.extend(masks.numpy().flatten())
        y_scores_pixel.extend(anomaly_maps.flatten())

    # Calculate AUROC scores
    image_auroc = roc_auc_score(y_true_img, y_scores_img) if len(np.unique(y_true_img)) > 1 else 0.5
    pixel_auroc = roc_auc_score(y_true_pixel, y_scores_pixel) if len(np.unique(y_true_pixel)) > 1 else 0.5

    return image_auroc, pixel_auroc