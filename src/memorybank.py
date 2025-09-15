import numpy as np
import faiss


class MemoryBank:
    def __init__(self, dim=128, use_gpu=False):
        """
        FAISS-based Memory Bank for anomaly detection.
        """
        self.dim = dim
        self.use_gpu = use_gpu
        self.index = None
        self.embeddings = []

    def add(self, embeddings):
        """
        Add embeddings to the memory pool.
        """
        if isinstance(embeddings, np.ndarray) is False:
            embeddings = embeddings.numpy()

        num_dims = embeddings.ndim
        if num_dims > 2:
            embeddings = embeddings.reshape(-1, embeddings.shape[-1])

        self.embeddings.append(embeddings.astype("float32"))

    def build(self, coreset_size=None):
        """
        Finalize the memory bank.
        """
        all_embeddings = np.concatenate(self.embeddings, axis=0)

        if coreset_size is not None and coreset_size < len(all_embeddings):
            selected = self._coreset_sampling(all_embeddings, coreset_size)
        else:
            selected = all_embeddings

        # Build FAISS index
        index = faiss.IndexFlatL2(self.dim)
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)

        index.add(selected)
        self.index = index
        self.embeddings = selected

        # --- FIX: Reset the list for the next epoch ---
        self.embeddings = []

    def query(self, queries, k=1):
        """
        Query anomaly scores for embeddings.
        """
        if self.index is None:
            raise ValueError("MemoryBank not built yet. Call .build() first.")

        if isinstance(queries, np.ndarray) is False:
            queries = queries.numpy()
        queries = queries.astype("float32")

        dists, _ = self.index.search(queries, k)
        return dists

    def _coreset_sampling(self, data, size):
        """
        Perform coreset sampling on the CPU to avoid OOM errors.
        """
        N, D = data.shape
        data = data.astype("float32")

        # Build a temporary FAISS index on the CPU
        index = faiss.IndexFlatL2(D)
        index.add(data)

        # Randomly pick first center
        idxs = [np.random.randint(N)]

        # Track min distances to selected centers
        min_dists = np.full(N, np.inf, dtype=np.float32)

        for _ in range(size - 1):
            last_center = data[idxs[-1]].reshape(1, -1)
            dists, _ = index.search(last_center, N)
            dists = dists.flatten()
            min_dists = np.minimum(min_dists, dists)

            next_idx = np.argmax(min_dists)
            idxs.append(next_idx)

        return data[idxs]