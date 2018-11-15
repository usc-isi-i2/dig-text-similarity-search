from typing import List, Union

__all__ = ['BaseProcessor']


class BaseProcessor(object):
    def __init__(self):
        self.indexer = None
        self.vectorizer = None
        self.index_builder = None 

    def vectorize(self, text: Union[str, List[str]]):
        raise NotImplementedError
