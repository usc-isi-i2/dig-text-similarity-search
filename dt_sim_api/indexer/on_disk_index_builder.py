import os
from typing import List

import faiss
import numpy as np

__all__ = ['OnDiskIndexBuilder']


class OnDiskIndexBuilder(object):
    """
    For building IVF index on-disk.
    Requires a pre-trained, empty index.
    """
    def __init__(self, path_to_empty_index):
        self.index = None
        self.invlist_paths = list()
        self.path_to_empty_index = path_to_empty_index

    def index_embeddings(self, embeddings: np.array, faiss_ids: np.array):
        assert embeddings.shape[0] == faiss_ids.shape[0]
        faiss_ids = np.reshape(faiss_ids, (faiss_ids.shape[0],))
        self.index.add_with_ids(embeddings, faiss_ids)

    @staticmethod
    def save_index(index, index_path: str):
        if not os.path.isfile(index_path) and index_path.endswith('.index'):
            try:
                faiss.write_index(index, index_path)
            except Exception as e:
                print(e)
                print('Could not save index')
        elif os.path.isfile(index_path) and index_path.endswith('.index'):
            print('Error: Index already exists: {}'.format(index_path))
        elif isinstance(index_path, str) and not index_path.endswith('.index'):
            print('Error: Index filename must end with .index \n'
                  '       Filename given: {}'.format(index_path.split('/')[-1]))
        else:
            print('Error: Unexpected path given {}'.format(index_path))

    def load_empty(self):
        empty_index = faiss.read_index(self.path_to_empty_index)
        if empty_index.is_trained and empty_index.ntotal == 0:
            self.index = empty_index
        else:
            raise Exception('Index must be empty and pre-trained.\n'
                            ' index.ntotal: ({}), index.is_trained: ({})'
                            ''.format(empty_index.ntotal, empty_index.is_trained))

    def generate_invlist(self, invlist_path, faiss_ids, embeddings: np.array):
        self.load_empty()
        self.index_embeddings(embeddings, faiss_ids)
        self.invlist_paths.append(invlist_path)
        self.save_index(self.index, invlist_path)
        del self.index
        self.index = None

    def n_invlists(self):
        print('* n invlists: {}'.format(len(self.invlist_paths)))

    def extend_invlist_paths(self, paths_to_add: List[str]):
        self.invlist_paths.extend(paths_to_add)
        self.n_invlists()

    def build_disk_index(self, merged_ivfs_path, merged_index_path) -> int:
        ivfs = list()
        for i, invlpth in enumerate(self.invlist_paths):
            index = faiss.read_index(invlpth, faiss.IO_FLAG_MMAP)
            ivfs.append(index.invlists)
            index.own_invlists = False      # Prevents de-allocation
            del index

        self.load_empty()
        invlists = faiss.OnDiskInvertedLists(self.index.nlist,
                                             self.index.code_size,
                                             merged_ivfs_path)

        ivf_vector = faiss.InvertedListsPtrVector()
        for ivf in ivfs:
            ivf_vector.push_back(ivf)

        ntotal = invlists.merge_from(ivf_vector.data(), ivf_vector.size())
        self.index.ntotal = ntotal
        self.index.replace_invlists(invlists)
        self.save_index(self.index, merged_index_path)
        self.index = None
        return int(ntotal)
