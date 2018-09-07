from typing import List
from .base_index_handler import *


class DeployIVF16K(BaseIndex):
    def __init__(self, path_to_deployable_index, nprobe: int = 32):
        BaseIndex.__init__(self)
        self.index = faiss.read_index(path_to_deployable_index)
        self.index.nprobe = nprobe

    def index_embeddings(self, embeddings: np.array, faiss_ids: np.array):
        self.index.add_with_ids(embeddings, faiss_ids)


class DiskBuildIVF16K(BaseIndex):
    def __init__(self, path_to_empty_index):
        BaseIndex.__init__(self)
        self.path_to_empty_index = path_to_empty_index
        self.invlist_paths = list()

    def index_embeddings(self, embeddings: np.array, faiss_ids: np.array):
        self.index.add_with_ids(embeddings, faiss_ids)

    def load_empty(self):
        empty_index = faiss.read_index(self.path_to_empty_index)
        if empty_index.is_trained and empty_index.ntotal == 0:
            self.index = empty_index
        else:
            raise Exception('Empty index ({}) must be pre-trained ({})'
                            ''.format(empty_index.ntotal, empty_index.is_trained))

    @staticmethod
    def generate_faiss_ids(npz_path, embeddings, sentences) -> np.array:
        """
        Generates unique faiss_ids by making an offset/tag from the npz_path

        Expected npz_path format:
            '/.../vectorized_new_<yyyy>-<mm>-<dd>_<fk>_10K<fk2>.npz'

        Offset/tag format: (21 digit np.long with 7 trailing zeros)
            <yyyy><mm><dd>0<fk><fk2>0000000

        :param npz_path: Path to preprocessed_news.npz
        :param embeddings: Preprocessed sentence embeddings in need of faiss_ids
        :param sentences: Corresponding sentences from news
        :return: Array of unique faiss_ids
        """
        assert len(embeddings) == len(sentences)

        filename = str(npz_path.split('/')[-1]).split('.')[0]
        date_keys = str(filename.split('_')[2]).split('-')
        file_keys = filename.split('_')[-2:]
        file_keys[1] = file_keys[1].split('K')[-1]

        offset = '{}{}{}0{}{}0000000'.format(date_keys[0], date_keys[1], date_keys[2],
                                             file_keys[0], file_keys[1])
        offset = np.long(offset)
        faiss_ids = offset + np.arange(start=0, stop=len(sentences), dtype=np.long)
        return faiss_ids

    def generate_invlists(self, npz_path, invlist_path,
                          embeddings: np.array, sentences: np.array):
        self.load_empty()
        fIDs = self.generate_faiss_ids(npz_path, embeddings, sentences)
        self.index_embeddings(embeddings, fIDs)
        self.invlist_paths.append(invlist_path)
        self.save_index(invlist_path)
        self.index = None

    def extend_invlist_paths(self, paths_to_add: List[str]):
        self.invlist_paths.extend(paths_to_add)
        print('* n invlists: {}'.format(len(self.invlist_paths)))

    def build_disk_index(self, merged_ivfs_path, merged_index_path):
        ivfs = list()
        for invlpth in self.invlist_paths:
            index = faiss.read_index(invlpth)
            ivfs.append(index.invlists)
            index.own_invlists = False      # Prevents de-allocation

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
