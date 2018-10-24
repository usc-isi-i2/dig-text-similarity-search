import os
import faiss
import numpy as np


class BaseIndex(object):
    def __init__(self):
        self.index = None

    @staticmethod
    def get_index_paths(index_dir_path):
        index_paths = list()
        for (dir_path, _, index_files) in os.walk(index_dir_path):
            for f in index_files:
                if f.endswith('.index'):
                    index_paths.append(os.path.join(dir_path, f))
            break
        return index_paths

    def index_embeddings(self, embeddings: np.array, faiss_ids: np.array):
        raise NotImplementedError

    def save_index(self, output_path):
        faiss.write_index(self.index, output_path)

    def search(self, query_vector: np.array, k: int):
        return self.index.search(query_vector, k)
