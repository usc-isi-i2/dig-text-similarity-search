from collections import OrderedDict
from typing import List, Tuple, Union

__all__ = ['BaseProcessor']


class BaseProcessor(object):
    def __init__(self):
        self.indexer = None
        self.vectorizer = None
        self.index_builder = None 

    def vectorize(self, text: Union[str, List[str]]):
        raise NotImplementedError

    @staticmethod
    def joint_sort(scores: List[List[float]], ids: List[List[int]]
                   ) -> Tuple[List[List[float]], List[List[int]]]:
        """
        Sorts scores in ascending order while maintaining score::id mapping.
        Checks if input is already sorted.
        :param scores: Faiss query/hit vector L2 distances
        :param ids: Corresponding faiss vector ids
        :return: Scores sorted in ascending order with corresponding ids
        """
        # Check
        if all(scores[0][i] <= scores[0][i+1] for i in range(len(scores[0])-1)):
            return scores, ids

        # Pythonic Joint Sort
        if isinstance(scores[0], list) and isinstance(ids[0], list):
            scores, ids = scores[0], ids[0]
        sorted_scores, sorted_ids = (list(sorted_scs_ids) for sorted_scs_ids
                                     in zip(*sorted(zip(scores, ids))))

        return [sorted_scores], [sorted_ids]
