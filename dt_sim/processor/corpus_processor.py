import os
import re
import os.path as p
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np

from .base_processor import *


__all__ = ['CorpusProcessor']


class CorpusProcessor(BaseProcessor):

    def __init__(self, vectorizer: object = None, index_builder: object = None,
                 progress_file: str = None):
        super().__init__()

        # Workhorses
        self.vectorizer = vectorizer
        self.index_builder = index_builder

        # Logging
        if progress_file:
            self.progress_file = p.abspath(progress_file)

    # Implement for BaseProcessor
    def batch_vectorize(self, text_batch: List[str], id_batch: List[str],
                        n_minibatch: int, very_verbose: bool = False
                        ) -> BatchReturn:
        assert len(text_batch) == len(id_batch)
        batched_embs = self.vectorizer.make_vectors(text_batch, n_minibatch,
                                                    verbose=very_verbose)

        batched_embs = np.vstack(batched_embs).astype(np.float32)
        batched_ids = np.array(id_batch, dtype=np.int64)

        return batched_embs, batched_ids

    # File Selection Funcs
    @staticmethod
    def track_preprocessing(progress_file: Union[str, Path],
                            verbose: bool = True) -> List[str]:
        preprocessed_news = list()
        if p.isfile(progress_file):
            with open(progress_file, 'r') as f:
                for line in f:
                    preprocessed_news.append(str(line).replace('\n', ''))
                    if verbose:
                        print(f'* Processed:  {preprocessed_news[-1]}')
        return sorted(preprocessed_news, reverse=True)

    @staticmethod
    def get_news_paths(news_dir: Union[str, Path], verbose: bool = True) -> List[Path]:
        assert p.isdir(news_dir), f'Error: Could not find {news_dir}'
        raw_news = list(Path(news_dir).glob('*.jl'))
        if verbose:
            for raw_file in raw_news:
                print(f'* Raw news:   {p.abspath(raw_file)}')

        return sorted(raw_news, reverse=True)

    @staticmethod
    def candidate_files(preprocessed_news: List[Union[str, Path]],
                        raw_news: List[Union[str, Path]],
                        verbose: bool = True) -> List[str]:
        files_to_process = list()
        for f in raw_news:
            if str(f) not in preprocessed_news:
                files_to_process.append(str(f))
                if verbose:
                    print(f'* Candidates: {str(f)}')
        return sorted(files_to_process, reverse=True)

    def select_file_to_process(self, news_dir: Union[str, Path],
                               verbose: bool = True) -> str:
        preprocessed_news = self.track_preprocessing(self.progress_file, verbose)
        all_news_paths = self.get_news_paths(news_dir, verbose)
        candidates = self.candidate_files(preprocessed_news, all_news_paths, verbose)
        candidates.sort(reverse=True)
        if verbose:
            print(f'Will process: {candidates[0]}\n')
        return candidates[0]

    def record_progress(self, preprocessed_file: Union[str, Path]):
        with open(self.progress_file, 'a') as prepped:
            prepped.write(f'{preprocessed_file}\n')

    # Path Funcs
    @staticmethod
    def init_paths(file_to_process: Union[str, Path],
                   seed: str = str('\d{4}[-/]\d{2}[-/]\d{2}')) -> Tuple[str, str]:
        """
        AIO func to initialize all directories necessary for shard preparation

        :param file_to_process: The ISO date will be extracted from the filename
            and incorporated into the path of new directories
        :param seed: Finds ISO date (i.e. YYYY-MM-DD)
        :return: Full path to subindex dir
        """
        try:
            date = re.search(seed, Path(file_to_process).stem).group()
        except AttributeError:
            raise Exception(f'Input filenames must contain their date'
                            f'formatted as YYYY-MM-DD. \n'
                            f'Input file given: {file_to_process}')

        tmp_idx_dir = p.abspath(p.join(Path(file_to_process).parent,
                                       '../tmp_idx_files/'))
        daily_dir = p.join(tmp_idx_dir, date)
        subidx_dir = p.join(daily_dir, 'subindexes')

        os.makedirs(tmp_idx_dir, exist_ok=True)
        os.makedirs(daily_dir, exist_ok=True)
        os.makedirs(subidx_dir, exist_ok=True)

        return subidx_dir, date
