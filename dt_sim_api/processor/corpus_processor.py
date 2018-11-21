import os
from typing import List, Union

from .base_processor import BaseProcessor
from dt_sim_api.indexer.on_disk_index_builder import OnDiskIndexBuilder
from dt_sim_api.vectorizer.sentence_vectorizer import SentenceVectorizer
from dt_sim_api.data_reader.jl_io_funcs import *
from dt_sim_api.data_reader.npz_io_funcs import *
from dt_sim_api.data_reader.misc_io_funcs import *


class CorpusProcessor(BaseProcessor):
    # TODO: move preprocessing scripts into methods
    # TODO: Add docstrings

    def __init__(self, vectorizer: object = None, index_builder: object = None,
                 large_USE: bool = False, empty_index_path: str = None):
        BaseProcessor.__init__(self)

        # TODO: Include new base indexes (when ready)
        if not empty_index_path:
            default_empty_index_path = 'dig-text-similarity-search/saved_indexes/' \
                                       'USE_lite_base_IVF16K.index'
            empty_index_path = os.path.abspath(default_empty_index_path)
            assert os.path.isfile(empty_index_path)

        if not index_builder:
            index_builder = OnDiskIndexBuilder(path_to_empty_index=empty_index_path)
        if not vectorizer:
            vectorizer = SentenceVectorizer(large=large_USE)

        self.index_builder = index_builder
        self.vectorizer = vectorizer

        ## I/O Funcs
        self.check_all_docs = check_all_docs
        self.get_all_docs = get_all_docs

        self.check_training_docs = check_training_docs
        self.get_training_docs = get_training_docs

        self.get_all_npz_paths = get_all_npz_paths
        self.load_training_npz = load_training_npz

        self.load_with_ids = load_with_ids
        self.save_with_ids = save_with_ids

        self.check_unique = check_unique
        self.clear_dir = clear_dir

    def vectorize(self, text: Union[str, List[str]]):
        return self.vectorizer.make_vectors(text)

    def include_subindex(self, subindex_paths: Union[str, List[str]]):
        self.index_builder.extend_subindex_paths(subindex_paths)
