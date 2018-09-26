import faiss
import numpy as np


class BaseIndex(object):
    def __init__(self):
        self.index = None

    def index_embeddings(self, embeddings: np.array, faiss_ids: np.array):
        raise NotImplementedError

    def save_index(self, output_path):
        faiss.write_index(self.index, output_path)

    def search(self, query_vector: np.array, k: int):
        return self.index.search(query_vector, k)
