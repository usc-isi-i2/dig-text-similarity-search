from time import sleep
from typing import List
from collections import OrderedDict
from multiprocessing import Pipe, Process, Queue
from .base_index_handler import *


class DeployIVF(BaseIndex):
    """
    For deploying on-disk index made with DiskBuilderIVF

    :param nprobe: Number of clusters to visit during search
        (speed accuracy trade-off)
    """
    def __init__(self, index_dir, nprobe: int = 32):
        BaseIndex.__init__(self)
        path_to_index = self.get_index_paths(index_dir)
        self.index = faiss.read_index(path_to_index[0])
        self.index.nprobe = nprobe

    def index_embeddings(self, embeddings: np.array, faiss_ids: np.array):
        print('WARNING: Cannot add to index \n'
              '   Hint: Use the DiskBuilderIVF class for adding to an index')


class DeployShards(BaseIndex):
    """
    For deploying multiple, pre-made IVF indexes as shards
        (intended for on-disk indexes that do not fit in memory)

    Note: The index shards must be true partitions with no overlapping ids

    :param paths_to_shards: List of paths to faiss index shards
    :param nprobe: Number of clusters to visit during search
        (speed accuracy trade-off)
    """
    def __init__(self, shard_dir, nprobe: int = 32):
        BaseIndex.__init__(self)
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

    def index_embeddings(self, embeddings: np.array, faiss_ids: np.array):
        print('WARNING: Cannot add to index shards \n'
              '   Hint: Use the DiskBuilderIVF class for adding to an index '
              'or self.add_shard(path) to add a new searchable shard')


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


class RangeShards(BaseIndex):

    def __init__(self, shard_dir, nprobe: int = 16, max_radius: float = 1.0):
        BaseIndex.__init__(self)
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
            D, I = D[0], I[0]

        D, I = self.sort(diff_scores=D, sent_ids=I)
        return [D], [I]

    @staticmethod
    def sort(diff_scores, sent_ids):
        results = dict()
        for score, sent_id in zip(diff_scores, sent_ids):
            if score not in results:
                results[score] = list()
            results[score].append(sent_id)

        sorted_results = OrderedDict(sorted(results.items()))
        for score in sorted_results:
            sorted_results[score].sort()

        sorted_scores = list()
        sorted_ids = list()
        for score, sids in sorted_results.items():
            for sent_id in sids:
                sorted_scores.append(score)
                sorted_ids.append(sent_id)
        return sorted_scores, sorted_ids

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

    def index_embeddings(self, embeddings: np.array, faiss_ids: np.array):
        print('WARNING: Cannot add to index shards \n'
              '   Hint: Use the DiskBuilderIVF class for adding to an index '
              'or self.add_shard(path) to add a new searchable shard')


#### IVF On-Disk Index Builder ################

class DiskBuilderIVF(BaseIndex):
    """
    For building IVF index on-disk.
    Requires a pre-trained, empty index.
    """
    def __init__(self, path_to_empty_index):
        BaseIndex.__init__(self)
        self.path_to_empty_index = path_to_empty_index
        self.invlist_paths = list()

    def index_embeddings(self, embeddings: np.array, faiss_ids: np.array):
        assert embeddings.shape[0] == faiss_ids.shape[0]
        faiss_ids = np.reshape(faiss_ids, (faiss_ids.shape[0],))
        self.index.add_with_ids(embeddings, faiss_ids)

    def load_empty(self):
        empty_index = faiss.read_index(self.path_to_empty_index)
        if empty_index.is_trained and empty_index.ntotal == 0:
            self.index = empty_index
        else:
            raise Exception('Index must be empty and pre-trained.\n'
                            ' index.ntotal: ({}), index.is_trained: ({})'
                            ''.format(empty_index.ntotal, empty_index.is_trained))

    def generate_invlist(self, invlist_path, faiss_ids,
                         embeddings: np.array) -> np.array:
        self.load_empty()
        self.index_embeddings(embeddings, faiss_ids)
        self.invlist_paths.append(invlist_path)
        self.save_index(invlist_path)
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
        self.save_index(merged_index_path)
        self.index = None
        return int(ntotal)
