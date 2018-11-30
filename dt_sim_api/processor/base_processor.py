from typing import List, Union

__all__ = ['BaseProcessor']


class BaseProcessor(object):
    def __init__(self):
        self.indexer = None
        self.vectorizer = None
        self.index_builder = None 

    def vectorize(self, text_batch: List[str], id_batch: List[str],
                  n_minibatch: int, very_verbose: bool = False):
        raise NotImplementedError
