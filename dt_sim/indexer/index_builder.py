import os
import os.path as p
import re
from time import time
from typing import List, Union

import numpy as np
import faiss

from dt_sim.data_reader.npz_io_funcs import load_training_npz

__all__ = ['OnDiskIVFBuilder']


class OnDiskIVFBuilder(object):
    """
    For building IVF indexes that do not fit in memory (searched on-disk).
    Building an on-disk index requires an empty, pre-trained base index.
    """

    def __init__(self, path_to_base_index: str):
        self.path_to_base_index = p.abspath(path_to_base_index)
        self.subindex_path_totals = dict()

    def zip_indexes(self, mv_dir: str, to_dir: str,
                    recursive: bool = False, mkdir: bool = False,
                    del_intermediates: bool = True):
        """
        Use this function to add freshly indexed news to an existing pub_date.index

        Naming Convention: Assumes all faiss.index and faiss.ivfdata files share
            the same name (i.e. they only differ by file extension)

        :param mv_dir: Move faiss.index & corresponding .ivfdata files from here
        :param to_dir: Zip with existing pub_date.index & .ivfdata files here
                    (groups by ISO publication date)
        :param recursive: Bool to search for nested faiss.index files in mv_dir
        :param mkdir: Bool to make to_dir if it does not exist
        :param del_intermediates: Bool to delete intermediate files
                    (if False, cp files without deleting)
        """
        t_start = time()

        mv_dir, to_dir = p.abspath(mv_dir), p.abspath(to_dir)
        if not p.isdir(to_dir) and mkdir:
            os.mkdir(to_dir)

        moving_indexes = self.find_indexes(mv_dir, recursive)
        target_indexes = self.find_indexes(to_dir)

        # Must be able to group multiple index paths by group_seed
        group_seed = str('\d{4}[-/]\d{2}[-/]\d{2}')     # ISO-date
        stale_files = list(moving_indexes)
        moving_groups = dict()
        while len(moving_indexes):
            index_path = moving_indexes.pop()
            check_date = re.search(group_seed, index_path).group()

            group = list()
            group.append(index_path)
            for idx_path in moving_indexes:
                if check_date in idx_path:
                    group.append(idx_path)
                    moving_indexes.pop(moving_indexes.index(idx_path))
            moving_groups[check_date] = group

        # Do not overwrite existing files
        tmp_indexes = list()
        tmp_dir = p.join(to_dir, 'tmp')
        for tgt_idx in target_indexes:
            for pub_date, _ in moving_groups.items():
                if pub_date in tgt_idx:
                    self.mv_index_and_ivfdata(
                        index_path=tgt_idx,
                        ivfdata_path=tgt_idx.replace('.index', '.ivfdata'),
                        to_dir=tmp_dir, mkdir=True
                    )
                    tmp_index_path = p.join(tmp_dir, tgt_idx.split('/')[-1])
                    tmp_indexes.append(tmp_index_path)
                    moving_groups[pub_date].append(tmp_index_path)

        # Merge moving & tmp faiss indexes in target dir
        for tgt_idx in target_indexes:
            for pub_date, group in moving_groups.items():
                if pub_date in tgt_idx:
                    self.merge_IVFs(
                        index_path=tgt_idx,
                        ivfdata_path=tgt_idx.replace('.index', '.ivfdata'),
                        ivfindex_paths=group
                    )

        # Delete intermediate files
        n_files = len(stale_files)
        n_existing = len(tmp_indexes)
        if del_intermediates:
            tmp_indexes.extend(stale_files)
            for tmp_idx in tmp_indexes:
                os.remove(tmp_idx)
                os.remove(tmp_idx.replace('.index', '.ivfdata'))
            os.rmdir(tmp_dir)

        print(f'\nMerged {n_files} file(s) with {n_existing} existing indexes '
              f'in {time()-t_start:0.2f}s')

    def mv_indexes(self, mv_dir: str, to_dir: str,
                   mkdir: bool = False, only_cp: bool = False):
        """
        Uses self.mv_index_and_ivfdata() to move (or copy) all on-disk, IVF
        indexes in mv_dir to to_dir.

        Note: Assumes filename.index and its corresponding filename.ivfdata
            only differ by file extension.

        DO NOT: $ mv my_faiss.index /new/dir/my_faiss.index or any parent dirs!
            The reference to its corresponding .ivfdata file will be lost!

        :param mv_dir: Move all faiss indexes (and .ivfdata files) from here ...
        :param to_dir: ... to this folder
        :param mkdir: Bool to make to_dir if it does not exist
        :param only_cp: Bool to prevent deletion of original files
                (i.e. $ mv ... acts like $ cp ...)
        """
        mv_dir, to_dir = p.abspath(mv_dir), p.abspath(to_dir)
        if not p.isdir(to_dir) and mkdir:
            os.mkdir(to_dir)

        moving_indexes = self.find_indexes(mv_dir)

        for idx in moving_indexes:
            self.mv_index_and_ivfdata(
                index_path=idx,
                ivfdata_path=idx.replace('.index', '.ivfdata'),
                to_dir=to_dir, mkdir=mkdir, only_cp=only_cp
            )

    def mv_index_and_ivfdata(self, index_path: str, ivfdata_path: str, to_dir: str,
                             mkdir: bool = False, only_cp: bool = False):
        """
        This function moves a specific on-disk faiss.index and its
        corresponding .ivfdata file into to_dir.

        DO NOT: $ mv my_faiss.index /new/dir/my_faiss.index or any parent dirs!
            The reference to its corresponding .ivfdata file will be lost!

        :param index_path: Move this faiss.index ...
        :param ivfdata_path: ... and this corresponding faiss.ivfdata ...
        :param to_dir: ... to this directory
        :param mkdir: Bool to make to_dir if it does not exist
        :param only_cp: Bool to prevent deletion of original files
                (i.e. $ mv ... acts like $ cp ...)
        """
        index_path, ivfdata_path = p.abspath(index_path), p.abspath(ivfdata_path)
        assert p.isfile(index_path), f'Could not find: {index_path}'
        assert p.isfile(ivfdata_path), f'Could not find: {ivfdata_path}'

        to_dir = p.abspath(to_dir)
        if not p.isdir(to_dir) and mkdir:
            os.mkdir(to_dir)

        if p.isdir(to_dir):
            new_index_path = p.join(to_dir, index_path.split('/')[-1])
            new_ivfdata_path = p.join(to_dir, ivfdata_path.split('/')[-1])
            assert not p.isfile(new_index_path) and not p.isfile(new_ivfdata_path), \
                f'Paths not clear! Check: {new_index_path} & {new_ivfdata_path}'

            n_vectors_mvd = self.merge_IVFs(
                index_path=p.abspath(new_index_path),
                ivfdata_path=p.abspath(new_ivfdata_path),
                ivfindex_paths=[index_path]
            )

            if only_cp:
                print(f'Copied: {index_path} and its .ivfdata file \n'
                      f'To:     {new_index_path} ({n_vectors_mvd} vectors)')
            else:
                os.remove(ivfdata_path), os.remove(index_path)
                print(f'Moved: {index_path} and its .ivfdata file \n'
                      f'To:    {new_index_path} ({n_vectors_mvd} vectors)')

        else:
            print(f'Unable to move index: {index_path} \n'
                  f'  * {to_dir} exists: {p.isdir(to_dir)} \n'
                  f'  * mkdir: {mkdir}')

    def merge_IVFs(self, index_path: str, ivfdata_path: str,
                   ivfindex_paths: List[str] = None) -> int:
        """
        An on-disk index must be built from existing subindexes. The
        inverted file list (IVF) from each subindex is merged into one
        disk-searchable .ivfdata file referenced by the .index file.

        Note: Use self.mv_index_and_ivfdata() to move these files.

        :param index_path: Path to output.index file
        :param ivfdata_path: Path to output.ivfdata file (on-disk searchable data)
        :param ivfindex_paths: Paths to indexes to be merged
        :return: Number of vectors indexed
        """
        # Collect IVF data from subindexes
        ivfs = list()
        if not ivfindex_paths:
            ivfindex_paths = list(self.subindex_path_totals.keys())
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
            index = self.index_embs_and_ids(index, embeddings, faiss_ids)
            self.subindex_path_totals[subindex_path] = index.ntotal
            faiss.write_index(index, subindex_path)

    @staticmethod
    def index_embs_and_ids(index, embeddings: np.array, faiss_ids: np.array):
        assert embeddings.shape[0] == faiss_ids.shape[0], \
            f'Found {embeddings.shape[0]} embeddings ' \
            f'and {faiss_ids.shape[0]} faiss_ids'
        faiss_ids = np.reshape(faiss_ids, (faiss_ids.shape[0],))
        index.add_with_ids(embeddings, faiss_ids)
        return index

    def include_subidx_path(self, paths_to_add: Union[str, List[str]]):
        """ Useful if subindexes already exist """
        if isinstance(paths_to_add, str):
            paths_to_add = [paths_to_add]
        for subidx_path in paths_to_add:
            if p.isfile(subidx_path) and subidx_path.endswith('.index'):
                n_vect = faiss.read_index(subidx_path).ntotal
                self.subindex_path_totals[subidx_path] = n_vect
            else:
                print(f'Unable to add index: {subidx_path}')
        self.print_n_subindexes()

    def print_n_subindexes(self):
        n_vectors = 0
        for subidx_path, n_vect in self.subindex_path_totals.items():
            n_vectors += n_vect
        print(f' {len(self.subindex_path_totals)} subindexes ({n_vectors} vectors)')

    def load_base_idx(self):
        base_index = faiss.read_index(self.path_to_base_index)
        if base_index.is_trained and base_index.ntotal == 0:
            return base_index
        else:
            raise Exception('Index must be empty and pre-trained.\n'
                            f'  * Index.ntotal: {base_index.ntotal} \n'
                            f'  * Index.is_trained: {base_index.is_trained}')

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

        index_type = f'{idx_type}{centroids},{compression}'
        print(f'\nCreating base faiss index: {index_type}')
        index = faiss.index_factory(dim, index_type)

        if not index.is_trained:
            print(' Training centroids...')
            t_train0 = time()
            index.train(training_set)
            print(f' Index trained in {time()-t_train0:0.2f}s')

        return index

    @staticmethod
    def write_index(index, save_path: str):
        print(' Saving trained base index...')
        faiss.write_index(index, save_path)
        print(f' Index saved as {save_path}')
        # TODO: index_metadata.txt

    @staticmethod
    def find_indexes(check_dir: str, recursive: bool = False):
        index_paths = list()
        for (parent_dir, nested_dirs, files) in os.walk(p.abspath(check_dir)):
            for f in files:
                if f.endswith('.index'):
                    index_paths.append(p.abspath(p.join(parent_dir, f)))
            if not recursive:
                break
        return index_paths

    @staticmethod
    def index_path_clear(index_path: str, file_suffix: str = '.index'):
        if not p.isfile(index_path) and index_path.endswith(file_suffix):
            return True
        elif p.isfile(index_path) and index_path.endswith(file_suffix):
            print(f'Index already exists: {index_path}')
            return False
        elif not index_path.endswith(file_suffix):
            print(f'Invalid index filename: {index_path}')
            return False
        else:
            print(f'Unexpected index path given: {index_path}')
            return False
