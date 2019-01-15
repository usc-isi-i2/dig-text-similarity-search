import os
from typing import List, Tuple

import numpy as np

from dt_sim.faiss_cache import faiss_cache

__all__ = ['BaseIndexer', 'faiss_cache',
           'DiffScores', 'VectorIDs', 'FaissSearch']


DiffScores = List[List[np.float32]]
VectorIDs = List[List[np.int64]]
FaissSearch = Tuple[DiffScores, VectorIDs]


class BaseIndexer(object):
    def __init__(self):
        self.index = None
        self.dynamic = False

    @faiss_cache(128)
    def search(self, query_vector: np.array, k: int) -> FaissSearch:
        return self.index.search(query_vector, k)

    @staticmethod
    def get_index_paths(index_dir_path):
        index_paths = list()
        for (dir_path, _, index_files) in os.walk(index_dir_path):
            for f in index_files:
                if f.endswith('.index'):
                    index_paths.append(os.path.join(dir_path, f))
            break
        return sorted(index_paths)

    @staticmethod
    def joint_sort(scores: DiffScores, ids: VectorIDs) -> FaissSearch:
        """
        Sorts scores in ascending order while maintaining score::id mapping.
        Checks if input is already sorted.
        :param scores: Faiss query/hit vector L2 distances
        :param ids: Corresponding faiss vector ids
        :return: Scores sorted in ascending order with corresponding ids
        """
        # Check if sorted
        if all(scores[0][i] <= scores[0][i + 1] for i in range(len(scores[0]) - 1)):
            return scores, ids

        # Possible nested lists
        if isinstance(scores[0], list) and isinstance(ids[0], list):
            scores, ids = scores[0], ids[0]

        # Joint sort
        sorted_scores, sorted_ids = (list(sorted_scs_ids) for sorted_scs_ids
                                     in zip(*sorted(zip(scores, ids))))
        return [sorted_scores], [sorted_ids]
