import tensorflow as tf
import faiss
import numpy as np


# =========================
# ⚡ FAISS Similarity Search
# =========================
def compute_similarity(z_bar_np, k=5, alpha=0.5, sigma=1.0):
    """
    Compute contextual + pairwise similarities with FAISS.
    Args:
        z_bar_np: numpy array of shape (B, N, D), teacher embeddings (float32).
    Returns:
        omega: (B, N, k) similarities (TF tensor)
        nn_idx: (B, N, k) neighbor indices (TF int32 tensor)
    """
    omega_batch = []
    nn_idx_batch = []

    for xb in z_bar_np:   # xb: (N, D)
        N, D = xb.shape

        # --- kNN search with FAISS ---
        index = faiss.IndexFlatL2(D)
        index.add(xb)
        dists, nn_idx = index.search(xb, k+1)  # include self
        dists, nn_idx = dists[:, 1:], nn_idx[:, 1:]  # drop self

        # --- Pairwise similarity ---
        omega_pairwise = np.exp(-dists / sigma)  # (N, k)

        # --- Contextual similarity ---
        context_mask = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            context_mask[i, nn_idx[i]] = 1.0

        overlap = context_mask @ context_mask.T
        denom = np.sum(context_mask, axis=-1, keepdims=True) + 1e-8
        omega_context_full = overlap / denom

        # shrink to neighbor subset
        omega_context = np.zeros_like(omega_pairwise)
        for i in range(N):
            omega_context[i] = omega_context_full[i, nn_idx[i]]

        # --- Blend ---
        omega_knn = alpha * omega_pairwise + (1 - alpha) * omega_context

        omega_batch.append(omega_knn)
        nn_idx_batch.append(nn_idx)

    return (
        tf.convert_to_tensor(np.stack(omega_batch), dtype=tf.float32),  # (B, N, k)
        tf.convert_to_tensor(np.stack(nn_idx_batch), dtype=tf.int32)    # (B, N, k)
    )


# =========================
# ⚡ Relaxed Contrastive Loss
# =========================

def relaxed_contrastive_loss(student_embeddings, omega, nn_idx, margin=1.0):
    """
    Relaxed contrastive loss (ReConPatch).
    Args:
        student_embeddings: (B, N, D) tensor
        omega: (B, N, k) similarities
        nn_idx: (B, N, k) neighbor indices
    Returns:
        loss: scalar tensor
    """
    def batch_loss(z, neigh, sim):
        # z: (N, D), neigh: (N, k), sim: (N, k)
        k = tf.shape(neigh)[1]
        z_i = tf.repeat(tf.expand_dims(z, 1), repeats=k, axis=1)  # (N, k, D)
        z_j = tf.gather(z, neigh)                                # (N, k, D)

        dists = tf.reduce_sum((z_i - z_j) ** 2, axis=-1)  # (N, k)
        mean_d = tf.reduce_mean(dists) + 1e-8
        delta = dists / mean_d

        pull = sim * (delta ** 2)                                # similar pairs
        push = (1.0 - sim) * tf.square(tf.nn.relu(margin - delta))  # dissimilar pairs
        return tf.reduce_mean(pull + push)

    # Map across batch dimension
    losses = tf.map_fn(
        lambda x: batch_loss(x[0], x[1], x[2]),
        (student_embeddings, nn_idx, omega),
        fn_output_signature=tf.float32,
    )
    return tf.reduce_mean(losses)
