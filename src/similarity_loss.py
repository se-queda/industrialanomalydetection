import tensorflow as tf
import faiss
import numpy as np


# =========================
# ⚡ FAISS Similarity Search
# =========================
def compute_similarity(z_bar_np, k=5, alpha=0.5, sigma=1.0):
    """
    Compute contextual + pairwise similarities with FAISS.
    """
    omega_batch = []
    nn_idx_batch = []

    for xb in z_bar_np:
        N, D = xb.shape
        index = faiss.IndexFlatL2(D)
        index.add(xb)
        dists, nn_idx = index.search(xb, k+1)
        dists, nn_idx = dists[:, 1:], nn_idx[:, 1:]

        omega_pairwise = np.exp(-dists / sigma)

        context_mask = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            context_mask[i, nn_idx[i]] = 1.0

        overlap = context_mask @ context_mask.T
        denom = np.sum(context_mask, axis=-1, keepdims=True) + 1e-8
        omega_context_full = overlap / denom

        omega_context = np.zeros_like(omega_pairwise)
        for i in range(N):
            omega_context[i] = omega_context_full[i, nn_idx[i]]

        omega_knn = alpha * omega_pairwise + (1 - alpha) * omega_context
        omega_batch.append(omega_knn)
        nn_idx_batch.append(nn_idx)

    return (
        np.stack(omega_batch).astype("float32"),
        np.stack(nn_idx_batch).astype("int32")
    )


# =========================
# ⚡ Relaxed Contrastive Loss
# =========================
@tf.function
def relaxed_contrastive_loss(student_embeddings, omega, nn_idx, margin=1.0):
    """
    Relaxed contrastive loss (ReConPatch).
    """
    def batch_loss(elems):
        z, neigh, sim = elems[0], elems[1], elems[2]
        k = tf.shape(neigh)[1]
        z_i = tf.repeat(tf.expand_dims(z, 1), repeats=k, axis=1)
        z_j = tf.gather(z, neigh)

        dists = tf.reduce_sum(tf.square(z_i - z_j), axis=-1)
        mean_d = tf.reduce_mean(dists) + 1e-8
        delta = dists / mean_d

        # Cast tensors to float32 before multiplication to avoid dtype mismatch
        sim = tf.cast(sim, tf.float32)
        delta = tf.cast(delta, tf.float32)

        pull = sim * tf.square(delta)
        push = (1.0 - sim) * tf.square(tf.nn.relu(margin - delta))
        return tf.reduce_mean(pull + push)

    losses = tf.map_fn(
        batch_loss,
        (student_embeddings, nn_idx, omega),
        fn_output_signature=tf.float32,
    )
    return tf.reduce_mean(losses)