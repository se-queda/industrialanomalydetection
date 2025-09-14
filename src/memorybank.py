import numpy as np
import faiss

class MemoryBank:
    def __init__(self, dim=128, use_gpu=False):
        """
        FAISS-based Memory Bank for anomaly detection.
        Args:
            dim: embedding dimension (default 128 from projectors).
            use_gpu: if True, FAISS runs on GPU.
        """
        self.dim = dim
        self.use_gpu = use_gpu
        self.index = None
        self.embeddings = []

    def add(self, embeddings):
        """
        Add embeddings to the memory pool (before coreset selection).
        Args:
            embeddings: numpy array of shape (N, D).
        """
        if isinstance(embeddings, np.ndarray) is False:
            embeddings = embeddings.numpy()
        self.embeddings.append(embeddings.astype("float32"))

    def build(self, coreset_size=None):
        """
        Finalize the memory bank.
        Args:
            coreset_size: if set, apply greedy coreset sampling to reduce.
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
        self.embeddings = selected  # store for reference

    def query(self, queries, k=1):
        """
        Query anomaly scores for embeddings.
        Args:
            queries: numpy array (M, D).
            k: nearest neighbors to retrieve.
        Returns:
            dists: (M, k) distances to nearest neighbors in memory.
        """
        if self.index is None:
            raise ValueError("MemoryBank not built yet. Call .build() first.")

        if isinstance(queries, np.ndarray) is False:
            queries = queries.numpy()
        queries = queries.astype("float32")

        dists, _ = self.index.search(queries, k)
        return dists

    def _coreset_sampling(self, data, size):
        N, D = data.shape
        data = data.astype("float32")

        # Build FAISS index for all embeddings
        index = faiss.IndexFlatL2(D)
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        index.add(data)

        # Randomly pick first center
        idxs = [np.random.randint(N)]

        # Track min distances to selected centers
        min_dists = np.full(N, np.inf, dtype=np.float32)

        for _ in range(size - 1):
            # Update distances to latest chosen center
            last_center = data[idxs[-1]].reshape(1, -1)
            dists, _ = index.search(last_center, N)  # distances from new center to all points
            dists = dists.flatten()
            min_dists = np.minimum(min_dists, dists)

            # Pick point farthest from any chosen center
            next_idx = np.argmax(min_dists)
            idxs.append(next_idx)

        return data[idxs]
