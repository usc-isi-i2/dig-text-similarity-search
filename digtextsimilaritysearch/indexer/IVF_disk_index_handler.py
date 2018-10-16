from typing import List
from .base_index_handler import *


class DeployIVF(BaseIndex):
    """
    For deploying on-disk index made with DiskBuilderIVF

    :param nprobe: Number of clusters to visit during search
        (speed accuracy trade-off)
    """
    def __init__(self, path_to_deployable_index, nprobe: int = 32):
        BaseIndex.__init__(self)
        self.index = faiss.read_index(path_to_deployable_index)
        self.index.nprobe = nprobe

    def index_embeddings(self, embeddings: np.array, faiss_ids: np.array):
        print('WARNING: Cannot add to index! \n'
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
    def __init__(self, paths_to_shards: List[str], nprobe: int = 32):
        BaseIndex.__init__(self)
        self.paths_to_shards = paths_to_shards
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
            print('WARNING: This shard is already online! \n'
                  '         Aborting...')
            return
        self.paths_to_shards.append(new_shard_path)
        shard = self.load_shard(path_to_shard=new_shard_path, nprobe=self.nprobe)
        self.index.add_shard(shard)

    def index_embeddings(self, embeddings: np.array, faiss_ids: np.array):
        print('WARNING: Cannot add to index shards! \n'
              '   Hint: Use the DiskBuilderIVF class for adding to an index '
              'or self.add_shard(path) to deploy a new shard')


class DeployDynamicShards(BaseIndex):
    """
    For deploying multiple, pre-made IVF indexes as shards dynamically,
    i.e. index shards are merged at query time.
        (intended for on-disk indexes that do not fit in memory)

    Note: The index shards must be true partitions with no overlapping ids

    :param paths_to_shards: List of paths to faiss index shards
    :param nprobe: Number of clusters to visit during search
        (speed accuracy trade-off)
    """
    def __init__(self, paths_to_shards: List[str], nprobe: int = 32):
        BaseIndex.__init__(self)
        self.paths_to_shards = paths_to_shards
        self.nprobe = nprobe
        self.dynamic = True

        # Load shards
        self.shards = list()
        for shard_path in self.paths_to_shards:
            shard_key, shard = self.load_shard(path_to_shard=shard_path, nprobe=self.nprobe)
            self.shards.append((shard_key, shard))
        self.shards.sort(key=lambda sk: sk[0], reverse=True)

    @staticmethod
    def load_shard(path_to_shard: str, nprobe: int = 32):
        shard_key = path_to_shard.split('/')[-1]
        shard = faiss.read_index(path_to_shard)
        shard.nprobe = nprobe
        return shard_key, shard

    def add_shard(self, new_shard_path: str):
        if new_shard_path in self.paths_to_shards:
            print('WARNING: This shard is already online! \n'
                  '         Aborting...')
            return
        self.paths_to_shards.append(new_shard_path)
        shard_key, shard = self.load_shard(path_to_shard=new_shard_path, nprobe=self.nprobe)
        self.shards.append((shard_key, shard))
        self.shards.sort(key=lambda sk: sk[0], reverse=True)

    def merge(self, s: int = 0, e: int = -1):
        for shard_tup in self.shards[s:e]:
            self.index.add_shard(shard_tup[1])

    def search(self, query_vector: np.array, k: int, start: int = 0, end: int = -1):
        self.index = faiss.IndexShards(512, threaded=True, successive_ids=False)
        self.merge(s=start, e=end)
        return next(self.yield_search(query_vector, k=k))

    def yield_search(self, query_vector: np.array, k: int):
        """
        Resets self.index in background after yielding search results.
        """
        yield self.index.search(query_vector, k=k)
        del self.index
        self.index = None

    def index_embeddings(self, embeddings: np.array, faiss_ids: np.array):
        print('WARNING: Cannot add to index shards! \n'
              '   Hint: Use the DiskBuilderIVF class for adding to an index '
              'or self.add_shard(path) to deploy a new shard')


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
