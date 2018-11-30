import os
import re
import datetime
import os.path as p
from time import time
from typing import List, Tuple, Union

import numpy as np

from .base_processor import BaseProcessor
from dt_sim_api.data_reader.jl_io_funcs import *
from dt_sim_api.data_reader.npz_io_funcs import *
from dt_sim_api.data_reader.misc_io_funcs import *
from dt_sim_api.indexer.index_builder import LargeIndexBuilder
from dt_sim_api.vectorizer.sentence_vectorizer import SentenceVectorizer


class CorpusProcessor(BaseProcessor):
    # TODO: move preprocessing scripts into methods
    # TODO: Add docstrings

    def __init__(self, vectorizer: object = None, index_builder: object = None,
                 progress_file: str = None):
        BaseProcessor.__init__(self)

        # Workhorses
        self.vectorizer = vectorizer
        self.index_builder = index_builder

        # Logging
        if progress_file:
            self.progress_file = p.abspath(progress_file)

    # Implement for BaseProcessor
    def vectorize(self, text_batch: List[str], id_batch: List[str],
                  n_minibatch: int, very_verbose: bool = False
                  ) -> Tuple[np.array, np.array]:
        assert len(text_batch) == len(id_batch)
        batched_embs = self.vectorizer.make_vectors(text_batch, n_minibatch,
                                                    verbose=very_verbose)

        batched_embs = np.vstack(batched_embs).astype(np.float32)
        batched_ids = np.array(id_batch, dtype=np.int64)

        return batched_embs, batched_ids

    # Preprocessing Methods

    # File Selection Funcs
    def record_progress(self, preprocessed_file: str):
        with open(self.progress_file, 'a') as prepped:
            prepped.write(preprocessed_file + '\n')

    @staticmethod
    def track_preprocessing(progress_file: str, verbose: bool = True) -> List[str]:
        preprocessed_news = list()
        if p.isfile(progress_file):
            with open(progress_file, 'r') as f:
                for line in f:
                    preprocessed_news.append(str(line).replace('\n', ''))
                    if verbose:
                        print('* Processed:  {}'.format(preprocessed_news[-1]))
        return sorted(preprocessed_news, reverse=True)

    @staticmethod
    def get_news_paths(news_dir: str, verbose: bool = True) -> List[str]:
        assert p.isdir(news_dir), 'Error: Could not find {}'.format(news_dir)
        raw_news = list()
        for (dir_path, _, file_list) in os.walk(news_dir):
            for f in file_list:
                if f.endswith('.jl'):
                    raw_news.append(str(p.join(dir_path, f)))
                    if verbose:
                        print('* Raw news:   {}'
                              ''.format(str(p.join(dir_path, f))))
            break
        return sorted(raw_news, reverse=True)

    @staticmethod
    def candidate_files(preprocessed_news: List[str], raw_news: List[str], 
                        verbose: bool = True) -> List[str]:
        files_to_process = list()
        for f in raw_news:
            if f not in preprocessed_news:
                files_to_process.append(str(f))
                if verbose:
                    print('* Candidates: {}'.format(str(f)))
        return sorted(files_to_process, reverse=True)

    def select_file_to_process(self, news_dir: str, verbose: bool = True) -> str:
        preprocessed_news = self.track_preprocessing(self.progress_file, verbose)
        all_news_paths = self.get_news_paths(news_dir, verbose)
        candidates = self.candidate_files(preprocessed_news, all_news_paths, verbose)
        candidates.sort(reverse=True)
        if verbose:
            print('Will process: {}\n'.format(candidates[0]))
        return candidates[0]

    # Path Funcs
    @staticmethod
    def init_paths(file_to_process: str, input_dir: str,
                   seed: str = str('\d{4}[-/]\d{2}[-/]\d{2}')) -> Tuple[str, str]:
        """
        TODO
        :param file_to_process:
        :param input_dir:
        :param seed: YYYY-MM-DD
        :return: Full path to subindex dir
        """
        try:
            date = re.search(seed, file_to_process).group()
        except AttributeError:
            raise Exception('Input filenames must contain their date'
                            'formatted as YYYY-MM-DD. \n'
                            'Input file given: {}'.format(file_to_process))
        tmp_idx_dir = p.abspath(p.join(input_dir, '../tmp_idx_files/'))
        daily_dir = p.join(tmp_idx_dir, date)
        subidx_dir = p.join(daily_dir, 'subindexes')
        if not p.isdir(tmp_idx_dir):
            os.mkdir(tmp_idx_dir)
        if not p.isdir(daily_dir):
            os.mkdir(daily_dir)
        if not p.isdir(subidx_dir):
            os.mkdir(subidx_dir)
        return subidx_dir, date
