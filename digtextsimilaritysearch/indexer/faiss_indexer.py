import faiss
from typing import List
import numpy as np


class FaissIndexer(object):
    def __init__(self, path_to_index_file=None):
        self.faiss_index = faiss.read_index(path_to_index_file) \
            if path_to_index_file \
            else faiss.IndexFlatL2(512)
            # else faiss.IndexIDMap(faiss.IndexFlatL2(512))

    def index_embeddings(self, embeddings) -> List[int]:
        ids = np.arange(len(embeddings)).astype('int64') + self.faiss_index.ntotal
        self.faiss_index.add(embeddings)
        return ids

    def save_index(self, output_path):
        faiss.write_index(self.faiss_index, output_path)

    def search(self, query_vector, k):
        return self.faiss_index.search(query_vector, k)
