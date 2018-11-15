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

    # noinspection PyTypeChecker
    @staticmethod
    def joint_sort(scores: List[List[float]], ids: List[List[int]]
                   ) -> Tuple[List[List[float]], List[List[int]]]:
        """
        Sorts scores in ascending order while maintaining score::id mapping. 
        Checks if input is already sorted. 
        
        :param scores: 
        :param ids: Corresponding ids
        :return: Scores sorted in ascending order with corresponding ids
        """
        # Check
        if all(scores[i] <= scores[i+1] for i in range(len(scores)-1)):
            return scores, ids

        # TODO: try zipped sort for more simplicity
        # Sort
        results = dict()
        for score, sent_id in zip(scores[0], ids[0]):
            if score not in results:
                results[score] = list()
            results[score].append(sent_id)

        consistent_results = OrderedDict(sorted(results.items()))
        for score in consistent_results:
            consistent_results[score].sort()

        sorted_scores = list()
        sorted_ids = list()
        for score, sids in consistent_results.items():
            for sent_id in sids:
                sorted_scores.append(score)
                sorted_ids.append(sent_id)

        return [sorted_scores], [sorted_ids]
