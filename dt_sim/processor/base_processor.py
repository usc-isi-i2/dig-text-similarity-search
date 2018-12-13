from typing import List, Tuple, Union
import numpy as np

__all__ = ['BaseProcessor', 'QueryReturn', 'BatchReturn']

QueryReturn = np.array
BatchReturn = Tuple[np.array, np.array]


class BaseProcessor(object):
    def __init__(self):
        self.indexer = None
        self.vectorizer = None
        self.index_builder = None 

    def vectorize(self, query: Union[str, List[str]]) -> QueryReturn:
        pass

    def batch_vectorize(self, text_batch: List[str], id_batch: List[str],
                        n_minibatch: int, very_verbose: bool = False
                        ) -> BatchReturn:
        pass
