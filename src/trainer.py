import tensorflow as tf
import numpy as np
import time
from tqdm import tqdm
from src.similarity_loss import compute_similarity, relaxed_contrastive_loss
from src.memorybank import MemoryBank
from src.patch_aggregator import aggregate


# =========================
# ⚡ Forward Pass
# =========================
@tf.function
def forward_pass(batch_images, backbone, f_proj, g_proj):
    """Backbone + projector forward pass."""
    feat_block2, feat_block3 = backbone(batch_images, training=False)
    patches_b2, patches_b3 = aggregate(feat_block2, feat_block3)
    z_f = f_proj(patches_b2)
    z_g = g_proj(patches_b3)
    return tf.concat([z_f, z_g], axis=1)


# =========================
# 🚀 Training Step
# =========================
@tf.function
def train_step(batch_images, backbone, f_proj, g_proj, student_model, ema_model, optimizer):
    """Training step for a single batch with mixed precision."""
    with tf.GradientTape() as tape:
        embeddings_from_proj = forward_pass(batch_images, backbone, f_proj, g_proj)
        student_embeddings = student_model(embeddings_from_proj, training=True)
        teacher_embeddings = ema_model(embeddings_from_proj, training=False)

        omega, nn_idx = tf.numpy_function(
            func=compute_similarity,
            inp=[teacher_embeddings],
            Tout=[tf.float32, tf.int32]
        )

        loss = relaxed_contrastive_loss(student_embeddings, omega, nn_idx, margin=1.0)
        scaled_loss = optimizer.get_scaled_loss(loss)

    trainable_vars = (
            f_proj.trainable_variables
            + g_proj.trainable_variables
            + student_model.trainable_variables
    )

    scaled_grads = tape.gradient(scaled_loss, trainable_vars)
    grads = optimizer.get_unscaled_gradients(scaled_grads)
    optimizer.apply_gradients(zip(grads, trainable_vars))
    ema_model.update_teacher()

    return loss, teacher_embeddings


def train_one_epoch(
        train_ds,
        backbone,
        f_proj,
        g_proj,
        student_model,
        ema_model,
        optimizer,
        memory_bank=None,
        coreset_size=None,
):
    epoch_loss = []

    for batch_images, _, _ in tqdm(train_ds, desc="Training"):
        loss, teacher_embeddings = train_step(
            batch_images, backbone, f_proj, g_proj, student_model, ema_model, optimizer
        )
        if memory_bank is not None:
            memory_bank.add(teacher_embeddings.numpy())
        epoch_loss.append(loss.numpy())

    if memory_bank is not None:
        memory_bank.build(coreset_size=coreset_size)

    return np.mean(epoch_loss)


# =========================
# 🏁 Training Driver
# =========================
def train(
        train_ds,
        backbone,
        f_proj,
        g_proj,
        student_model,
        ema_model,
        optimizer,
        epochs=10,
        coreset_size=10000,
):
    memory_bank = MemoryBank(dim=128, use_gpu=True)

    for epoch in range(epochs):
        start_time = time.time()
        print(f"\n--- EPOCH {epoch + 1}/{epochs} ---")

        avg_loss = train_one_epoch(
            train_ds,
            backbone,
            f_proj,
            g_proj,
            student_model,
            ema_model,
            optimizer,
            memory_bank=memory_bank,
            coreset_size=coreset_size,
        )

        epoch_time = time.time() - start_time
        print(f"Epoch training time: {epoch_time:.2f} seconds | Average Loss = {avg_loss:.4f}")

    return memory_bank