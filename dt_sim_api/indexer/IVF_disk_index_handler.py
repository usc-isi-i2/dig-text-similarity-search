import os
from time import sleep
from typing import List, Tuple
from multiprocessing import Pipe, Process, Queue

import faiss
import numpy as np

from .base_indexer import BaseIndexer


class DeployIVF(BaseIndexer):
    """
    For deploying on-disk index made with DiskBuilderIVF

    :param nprobe: Number of clusters to visit during search
        (speed accuracy trade-off)
    """
    def __init__(self, index_dir, nprobe: int = 32):
        BaseIndexer.__init__(self)
        path_to_index = self.get_index_paths(index_dir)
        self.index = faiss.read_index(path_to_index[0])
        self.index.nprobe = nprobe


class DeployShards(BaseIndexer):
    """
    For deploying multiple, pre-made IVF indexes as shards
        (intended for on-disk indexes that do not fit in memory)

    Note: The index shards must be true partitions with no overlapping ids

    :param paths_to_shards: List of paths to faiss index shards
    :param nprobe: Number of clusters to visit during search
        (speed accuracy trade-off)
    """
    def __init__(self, shard_dir, nprobe: int = 32):
        BaseIndexer.__init__(self)
        self.paths_to_shards = self.get_index_paths(shard_dir)
        self.nprobe = nprobe

        # Load shards
        shards = list()
        for shard_path in self.paths_to_shards:
            shard = self.load_shard(path_to_shard=shard_path, nprobe=self.nprobe)
            shards.append(shard)

        # Merge shards
        self.index = faiss.IndexShards(512, threaded=True, successive_ids=False)
        for shard in shards:
            self.index.add_shard(shard)

    @staticmethod
    def load_shard(path_to_shard: str, nprobe: int = 32):
        shard = faiss.read_index(path_to_shard)
        shard.nprobe = nprobe
        return shard

    def add_shard(self, new_shard_path: str):
        if new_shard_path in self.paths_to_shards:
            print('WARNING: This shard is already online \n'
                  '         Aborting...')
            return
        self.paths_to_shards.append(new_shard_path)
        shard = self.load_shard(path_to_shard=new_shard_path, nprobe=self.nprobe)
        self.index.add_shard(shard)


#### Parallel Range Search ####################

class Shard(Process):

    def __init__(self, shard_name, shard_path, input_pipe, output_queue,
                 nprobe: int = 16, daemon: bool = False):
        Process.__init__(self, name=shard_name)
        self.daemon = daemon

        self.input = input_pipe
        self.index = faiss.read_index(shard_path)
        self.index.nprobe = nprobe
        self.output = output_queue

    def run(self):
        if self.input.poll():
            (query_vector, radius) = self.input.recv()
            _, dd, ii = self.index.range_search(query_vector, radius)
            self.output.put((dd, ii), block=False)


class RangeShards(BaseIndexer):

    def __init__(self, shard_dir, nprobe: int = 16, max_radius: float = 1.0):
        BaseIndexer.__init__(self)
        self.paths_to_shards = self.get_index_paths(shard_dir)
        self.nprobe = nprobe
        self.max_radius = max_radius
        self.dynamic = False        # TODO: Coordinate with frontend for date-range search
        self.lock = False

        self.results = Queue()
        self.shards = dict()
        for shard_path in self.paths_to_shards:
            self.load_shard(shard_path)
        for shard_name, (handler_pipe, shard) in self.shards.items():
            shard.start()            

    def load_shard(self, shard_path):
        shard_name = shard_path.replace('.index', '').split('/')[-1]
        shard_pipe, handler_pipe = Pipe(False)
        shard = Shard(shard_name, shard_path,
                      input_pipe=shard_pipe, output_queue=self.results,
                      nprobe=self.nprobe, daemon=False)
        self.shards[shard_name] = (handler_pipe, shard)

    def search(self, query_vector: np.array, k: int, radius: float = 0.5):
        if len(query_vector.shape) < 2 or query_vector.shape[0] > 1:
            query_vector = np.reshape(query_vector, (1, query_vector.shape[0]))

        # Lock search while loading index
        if self.lock:
            sleep(0.25)
            return self.search(query_vector, k, radius)

        # Start parallel range search
        for shard_name, (hpipe, shard) in self.shards.items():
            hpipe.send((query_vector, radius))
            shard.run()

        # Aggregate results
        D, I = list(), list()
        while not self.results.empty():
            dd, ii = self.results.get()
            D.extend(dd), I.extend(ii)

        # Ensure len(results) > k
        if radius < self.max_radius and len(D) < k:
            new_radius = radius + 0.3
            D, I = self.search(query_vector, k, radius=new_radius)
        return self.joint_sort(D, I)

    def add_shard(self, new_shard_path: str):
        # Lock search while loading shard
        self.lock = True

        shard_name = new_shard_path.replace('.index', '').split('/')[-1]
        if new_shard_path in self.paths_to_shards or shard_name in self.shards:
            print('WARNING: This shard is already online \n'
                  '         Aborting...')
        else:
            self.paths_to_shards.append(new_shard_path)
            self.load_shard(new_shard_path)

        # Release lock
        sleep(0.05)
        self.lock = False


#### IVF On-Disk Index Builder ################

class DiskBuilderIVF(BaseIndexer):
    """
    For building IVF index on-disk.
    Requires a pre-trained, empty index.
    """
    def __init__(self, path_to_empty_index):
        BaseIndexer.__init__(self)
        self.path_to_empty_index = path_to_empty_index
        self.invlist_paths = list()

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
            print('Error: Index already exists: {} \n'
                  '       Aborting...'.format(index_path))
        elif isinstance(index_path, str) and not index_path.endswith('.index'):
            print('Error: Index filename must end with .index \n'
                  '       Filename given: {} \n'
                  '       Aborting...'.format(index_path.split('/')[-1]))
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
