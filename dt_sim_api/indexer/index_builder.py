import os
import os.path as p
from typing import List, Union

import numpy as np
import faiss

from dt_sim_api.data_reader.npz_io_funcs import load_training_npz

__all__ = ['LargeIndexBuilder']


class LargeIndexBuilder(object):
    """
    For building IVF indexes that do not fit in memory (searched on-disk).
    Building an on-disk index equires an empty, pre-trained base index.
    """
    def __init__(self, path_to_base_index: str):
        self.path_to_base_index = p.abspath(path_to_base_index)
        self.subindex_paths = list()

    def mv_index_and_ivfdata(self, index_path: str, ivfdata_path: str, new_dir: str):
        """ Use this function for moving on-disk indexes (DO NOT: $ mv ...) """
        index_path, ivfdata_path = p.abspath(index_path), p.abspath(ivfdata_path)
        assert p.isfile(index_path), 'Could not find: {}'.format(index_path)
        assert p.isfile(ivfdata_path), 'Could not find: {}'.format(ivfdata_path)

        new_index_path = p.abspath(p.join(new_dir, index_path.split('/')[-1]))
        new_ivfdata_path = p.abspath(p.join(new_dir, ivfdata_path.split('/')[-1]))
        n_vectors_mvd = self.merge_IVFs(new_index_path, new_ivfdata_path,
                                        ivfindex_paths=[index_path])

        os.remove(ivfdata_path), os.remove(index_path)
        print('Moved: {} and its .ivfdata file \n'
              'To:    {} ({} total vectors)'
              ''.format(index_path, new_index_path, n_vectors_mvd))

    def merge_IVFs(self, index_path: str, ivfdata_path: str,
                   ivfindex_paths: List[str] = None) -> int:
        """
        An on-disk index must be built from existing subindexes. The
        inverted file list (IVF) from each subindex is merged into one
        disk-searchable .ivfdata file referenced by the .index file.

        Note: Use self.mv_index_and_ivfdata() to move these files.

        :param ivfdata_path: Path to on-disk searchable IVF data
        :param index_path: Path to .index file (load this)
        :param ivfindex_paths: 
        :return: Number of vectors indexed
        """
        assert self.index_path_clear(index_path)
        assert self.index_path_clear(ivfdata_path, '.ivfdata')

        # Collect IVF data from subindexes
        ivfs = list()
        if not ivfindex_paths:
            ivfindex_paths = self.subindex_paths
        for subindex_path in ivfindex_paths:
            index = faiss.read_index(subindex_path, faiss.IO_FLAG_MMAP)
            ivfs.append(index.invlists)
            index.own_invlists = False  # Prevents de-allocation
            del index

        # Prepare .ivfdata file
        index = self.load_base_idx()
        invlists = faiss.OnDiskInvertedLists(index.nlist,
                                             index.code_size,
                                             ivfdata_path)
        ivf_vector = faiss.InvertedListsPtrVector()
        for ivf in ivfs:
            ivf_vector.push_back(ivf)

        # Merge IVF data
        ntotal = invlists.merge_from(ivf_vector.data(), ivf_vector.size())
        index.ntotal = ntotal
        index.replace_invlists(invlists)
        faiss.write_index(index, index_path)
        return int(ntotal)

    def generate_subindex(self, subindex_path: str, 
                          embeddings: np.array, faiss_ids: np.array):
        if self.index_path_clear(subindex_path):
            index = self.load_base_idx()
            self.index_embs_and_ids(index, embeddings, faiss_ids)
            self.subindex_paths.append(subindex_path)
            faiss.write_index(index, subindex_path)

    @staticmethod
    def index_embs_and_ids(index, embeddings: np.array, faiss_ids: np.array):
        assert embeddings.shape[0] == faiss_ids.shape[0], \
            'Found {} embeddings and {} faiss_ids' \
            ''.format(embeddings.shape[0], faiss_ids.shape[0])
        faiss_ids = np.reshape(faiss_ids, (faiss_ids.shape[0],))
        index.add_with_ids(embeddings, faiss_ids)
        return index

    def extend_subindex_paths(self, paths_to_add: Union[str, List[str]]):
        """ Useful if subindexes already exist """
        if isinstance(paths_to_add, str):
            paths_to_add = [paths_to_add]
        for subindex_path in paths_to_add:
            if p.isfile(subindex_path) and subindex_path.endswith('.index'):
                self.subindex_paths.append(subindex_path)
        self.print_n_subindexes()

    def print_n_subindexes(self):
        n_vectors = 0
        for subidx in self.subindex_paths:
            n_vectors += faiss.read_index(subidx).ntotal
        print('N subindexes: {} ({} total vectors)'
              ''.format(len(self.subindex_paths), n_vectors))

    def load_base_idx(self):
        base_index = faiss.read_index(self.path_to_base_index)
        if base_index.is_trained and base_index.ntotal == 0:
            return base_index
        else:
            raise Exception('Index must be empty and pre-trained.\n'
                            'Index.ntotal: ({}), Index.is_trained: ({})'
                            ''.format(base_index.ntotal, base_index.is_trained))

    def setup_base_index(self, centroids: int, ts_path: str,
                         npz_dir: str = None, n_tr_vectors: int = 1000000,
                         idx_type: str = 'IVF', compression: str = 'Flat',
                         dim: int = 512, base_index_path: str = None):
        # TODO: Docstring
        if not base_index_path:
            base_index_path = self.path_to_base_index
        assert self.index_path_clear(base_index_path)

        tr_vectors = load_training_npz(ts_path, npz_dir, n_vectors=n_tr_vectors)
        base_index = self.make_base_index(idx_type, centroids, compression,
                                          training_set=tr_vectors, dim=dim)

        self.write_index(index=base_index, save_path=base_index_path)

    @staticmethod
    def make_base_index(idx_type: str, centroids: int, compression: str,
                        training_set: np.ndarray, dim: int = 512):

        index_type = '{}{},{}'.format(idx_type, centroids, compression)
        print('\nCreating base faiss index: {}'.format(index_type))
        index = faiss.index_factory(dim, index_type)

        if not index.is_trained:
            print(' Training centroids...')
            t_train0 = time()
            index.train(training_set)
            print(' Index trained in {:0.2f}s'.format(time() - t_train0))

        return index

    @staticmethod
    def write_index(index, save_path: str):
        print(' Saving trained base index...')
        faiss.write_index(index, save_path)
        print(' Index saved as {}'.format(save_path))
        # TODO: index_metadata.txt

    @staticmethod
    def index_path_clear(index_path: str, file_suffix: str = '.index'):
        if not p.isfile(index_path) and index_path.endswith(file_suffix):
            return True
        elif p.isfile(index_path) and index_path.endswith(file_suffix):
            print('Index already exists: {}'.format(index_path))
            return False
        elif not index_path.endswith(file_suffix):
            print('Invalid index filename: {}'.format(index_path))
            return False
        else:
            print('Unexpected index path given: {}'.format(index_path))
            return False
