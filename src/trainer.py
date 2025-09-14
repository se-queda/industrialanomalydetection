import tensorflow as tf
import numpy as np
from src.similarity_loss import compute_similarity, relaxed_contrastive_loss
from src.memorybank import MemoryBank
from src.patch_aggregator import aggregate


# =========================
# ⚡ Forward Pass (XLA JIT)
# =========================

def forward_pass(batch_images, backbone, f_proj, g_proj):
    """Backbone + projector forward pass."""
    feat_block2, feat_block3 = backbone(batch_images, training=False)
    patches_b2, patches_b3 = aggregate(feat_block2, feat_block3)
    z_f = f_proj(patches_b2)  # (B, n_b2, 128)
    z_g = g_proj(patches_b3)  # (B, n_b3, 128)
    return tf.concat([z_f, z_g], axis=1)  # (B, N, 128)


# =========================
# 🚀 One Epoch
# =========================
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
    log_interval=50,
):
    epoch_loss = []
    step = 0

    for batch_images, _ in train_ds:
        with tf.GradientTape() as tape:
            # --- Forward pass (inside tape so grads flow) ---
            student_embeddings = forward_pass(batch_images, backbone, f_proj, g_proj)

            # --- Teacher embeddings (NumPy for FAISS) ---
            teacher_embeddings = (
                ema_model(student_embeddings, training=False)
                .numpy()
                .astype("float32")
            )

            # --- Similarity + Loss ---
            omega, nn_idx = compute_similarity(teacher_embeddings, k=5)
            loss = relaxed_contrastive_loss(student_embeddings, omega, nn_idx, margin=1.0)

        # --- Backprop ---
        grads = tape.gradient(
            loss,
            f_proj.trainable_variables
            + g_proj.trainable_variables
            + student_model.trainable_variables,
        )
        optimizer.apply_gradients(
            zip(
                grads,
                f_proj.trainable_variables
                + g_proj.trainable_variables
                + student_model.trainable_variables,
            )
        )

        # --- EMA teacher update ---
        ema_model.update_teacher()

        # --- Memory bank update ---
        if memory_bank is not None:
            memory_bank.add(teacher_embeddings)

        # --- Logging ---
        epoch_loss.append(loss.numpy())
        if step % log_interval == 0:
            tf.print("[STEP", step, "] Loss =", loss)

        step += 1

    # --- Finalize memory bank ---
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
    epochs=10,
    lr=1e-3,
    coreset_size=10000,
):
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    memory_bank = MemoryBank(dim=128, use_gpu=True)

    for epoch in range(epochs):
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
            log_interval=20,
        )
        print(f"[EPOCH {epoch+1}] Average Loss = {avg_loss:.4f}")

    return memory_bank
