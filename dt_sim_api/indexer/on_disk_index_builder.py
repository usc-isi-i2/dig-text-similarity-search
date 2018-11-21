import os
from typing import List, Union

import faiss
import numpy as np

__all__ = ['OnDiskIndexBuilder']


class OnDiskIndexBuilder(object):
    def __init__(self, path_to_empty_index):
        """
        For building IVF indexes that are searchable on-disk.
        :param path_to_empty_index: Path to empty, pre-trained IVF index
        """
        self.index = None
        self.subindex_paths = list()
        self.path_to_empty_index = path_to_empty_index

    def merge_IVFs(self, merged_ivfs_path: str, merged_index_path: str) -> int:
        """
        An on-disk index must be built from existing subindexes. The
        inverted file list (ivf) from each subindex is merged into one
        disk-searchable .ivfdata file referenced by the .index file.

        Note: Once built, on-disk indexes cannot move directories.

        :param merged_ivfs_path: Path to on-disk searchable IVF data
        :param merged_index_path: Path to .index file (load this)
        """
        if not self.index_path_clear(merged_ivfs_path, '.ivfdata'):
            print('Error: Cannot build .ivfdata file')
            return 0
        if not self.index_path_clear(merged_index_path):
            print('Error: Cannot build .index file')
            return 0

        # Collect IVF data from subindexes
        ivfs = list()
        for i, subindex_path in enumerate(self.subindex_paths):
            index = faiss.read_index(subindex_path, faiss.IO_FLAG_MMAP)
            ivfs.append(index.invlists)
            index.own_invlists = False  # Prevents de-allocation
            del index

        # Prepare .ivfdata file
        self.load_empty()
        invlists = faiss.OnDiskInvertedLists(self.index.nlist,
                                             self.index.code_size,
                                             merged_ivfs_path)
        ivf_vector = faiss.InvertedListsPtrVector()
        for ivf in ivfs:
            ivf_vector.push_back(ivf)

        # Merge IVF data
        ntotal = invlists.merge_from(ivf_vector.data(), ivf_vector.size())
        self.index.ntotal = ntotal
        self.index.replace_invlists(invlists)
        faiss.write_index(self.index, merged_index_path)
        self.index = None
        return int(ntotal)

    def generate_subindex(self, subindex_path: str, 
                          embeddings: np.array, faiss_ids: np.array):
        if self.index_path_clear(subindex_path):
            self.load_empty()
            self.index_embeddings(embeddings, faiss_ids)
            self.subindex_paths.append(subindex_path)
            faiss.write_index(self.index, subindex_path)
            self.index = None

    def load_empty(self):
        empty_index = faiss.read_index(self.path_to_empty_index)
        if empty_index.is_trained and empty_index.ntotal == 0:
            self.index = empty_index
        else:
            raise Exception('Index must be empty and pre-trained.\n'
                            'Index.ntotal: ({}), Index.is_trained: ({})'
                            ''.format(empty_index.ntotal, empty_index.is_trained))

    def index_embeddings(self, embeddings: np.array, faiss_ids: np.array):
        assert embeddings.shape[0] == faiss_ids.shape[0], \
            'Found {} embeddings and {} faiss_ids' \
            ''.format(embeddings.shape[0], faiss_ids.shape[0])
        faiss_ids = np.reshape(faiss_ids, (faiss_ids.shape[0],))
        self.index.add_with_ids(embeddings, faiss_ids)

    def extend_subindex_paths(self, paths_to_add: Union[str, List[str]]):
        """ Useful if subindexes already exist """
        if isinstance(paths_to_add, str):
            paths_to_add = [paths_to_add]
        for subindex_path in paths_to_add:
            if os.path.isfile(subindex_path) and subindex_path.endswith('.index'):
                self.subindex_paths.append(subindex_path)
        self.print_n_subindexes()

    def print_n_subindexes(self):
        print('Number of subindexes to merge: {}'.format(len(self.subindex_paths)))

    @staticmethod
    def index_path_clear(index_path: str, file_suffix: str = '.index'):
        if not os.path.isfile(index_path) and index_path.endswith(file_suffix):
            return True
        elif os.path.isfile(index_path) and index_path.endswith(file_suffix):
            print('Error: Index already exists: {}'.format(index_path))
            return False
        elif not index_path.endswith(file_suffix):
            print('Error: Invalid index filename: {}'.format(index_path))
            return False
        else:
            print('Error: Unexpected index path given: {}'.format(index_path))
            return False
