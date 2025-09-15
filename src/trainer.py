import tensorflow as tf
import numpy as np
import time
import os
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

    return loss


def train_one_epoch(
        train_ds,
        backbone,
        f_proj,
        g_proj,
        student_model,
        ema_model,
        optimizer,
):
    epoch_loss = []
    for batch_images, _, _ in tqdm(train_ds, desc="Training"):
        loss = train_step(
            batch_images, backbone, f_proj, g_proj, student_model, ema_model, optimizer
        )
        epoch_loss.append(loss.numpy())

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
        checkpoint_dir="checkpoints"
):
    # --- CHECKPOINTING SETUP ---
    ckpt = tf.train.Checkpoint(student_model=student_model,
                               ema_model=ema_model,
                               optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=5)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
        print(f"\n[INFO] Restored from checkpoint: {ckpt_manager.latest_checkpoint}")
    else:
        print("\n[INFO] Initializing from scratch.")

    # --- MAIN TRAINING LOOP ---
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
        )

        save_path = ckpt_manager.save()
        print(f"Epoch training time: {time.time() - start_time:.2f} seconds | Average Loss = {avg_loss:.4f}")
        print(f"Checkpoint saved at: {save_path}")

    print("\n[INFO] All training epochs complete. Building the final memory bank...")

    # --- FINAL MEMORY BANK BUILD (after all training) ---
    memory_bank = MemoryBank(dim=128, use_gpu=True)
    for batch_images, _, _ in tqdm(train_ds, desc="Building Memory Bank"):
        teacher_embeddings = ema_model(forward_pass(batch_images, backbone, f_proj, g_proj), training=False)
        memory_bank.add(teacher_embeddings.numpy())

    memory_bank.build(coreset_size=coreset_size)
    print("[INFO] Memory bank built successfully.")

    return memory_bank