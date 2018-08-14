import os

import faiss
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from copy import deepcopy


class Base(object):

    def __init__(self,
                 index_dir: str,
                 grand_index_name: str = None,
                 base_index_name: str = None,
                 path_to_model: str = None
                 ) -> None:

        assert os.path.isdir(index_dir)
        self._index_dir = index_dir

        if not grand_index_name.endswith(".index"):
            grand_index_name += ".index"
        self._path_to_grand_index = os.path.join(index_dir, grand_index_name)
        self._grand_index = None

        if not base_index_name.endswith(".index"):
            base_index_name += ".index"
        self._path_to_base_index = os.path.join(index_dir, base_index_name)
        self._base_index = None

        self._path_to_model = path_to_model
        self._model = None

        self._sess_active = False
        self._query_mode = False
        self._batch_mode = False
        self._sess = None

    @staticmethod
    def initialize_index(factory_str: str = None) -> faiss.Index:
        if factory_str:
            index = faiss.index_factory(512, factory_str)
        else:
            index = faiss.IndexFlatL2(512)
        return index

    @property
    def index_dir(self) -> os.path:
        return self._index_dir

    @property
    def path_to_grand_index(self) -> os.path:
        return self._path_to_grand_index

    @path_to_grand_index.setter
    def path_to_grand_index(self, new_path) -> None:
        assert os.path.exists(new_path)
        self._path_to_grand_index = new_path

    @property
    def grand_index(self) -> faiss.Index:
        if not self._grand_index:
            if not os.path.exists(self.path_to_grand_index):
                self._grand_index = deepcopy(self.base_index)
            else:
                self._grand_index = faiss.read_index(self.path_to_grand_index)
                print("Grand index loaded from {}".format(self.path_to_grand_index))
            print("  Index is trained: {}".format(self._grand_index.is_trained))
        return self._grand_index

    def set_grand_index(self, new_index: faiss.Index) -> None:
        assert isinstance(new_index, faiss.Index), "Only trained faiss.Index(es) " \
                                                   "can be assigned to the " \
                                                   "grand index"
        try:
            assert new_index.is_trained, "New index is not trained."
            self._grand_index = deepcopy(new_index)
            print("Grand index is now {}".format(type(self._grand_index)))
            print("  Index is trained: {}".format(self._grand_index.is_trained))
        except AssertionError:
            print("Send to base index instead.")

    def save_grand_index(self, grand_index_savename: str) -> None:
        if not grand_index_savename.endswith(".index"):
            grand_index_savename += ".index"
        faiss.write_index(self.grand_index, grand_index_savename)

    @property
    def path_to_base_index(self) -> os.path:
        return self._path_to_base_index

    @path_to_base_index.setter
    def path_to_base_index(self, new_path) -> None:
        assert os.path.exists(new_path)
        self._path_to_base_index = new_path

    @property
    def base_index(self) -> faiss.Index:
        if not self._base_index:
            if not os.path.exists(self.path_to_base_index):
                self._base_index = self.initialize_index()
            else:
                self._base_index = faiss.read_index(self.path_to_base_index)
                print("Base index loaded from {}".format(self.path_to_base_index))
            print("  Index is trained: {}".format(self._grand_index.is_trained))
        return self._base_index

    def set_base_index(self, new_index: str or faiss.Index = None) -> None:
        if isinstance(new_index, str):
            self._base_index = faiss.index_factory(512, new_index)
        else:
            assert isinstance(new_index, faiss.Index), "Cannot make new index " \
                                                       "from {}".format(new_index)
            self._base_index = deepcopy(new_index)
        print("Base index is now {}".format(type(new_index)))
        print("  Index is trained: {}".format(self._base_index.is_trained))

    def save_base_index(self, base_index_savename: str) -> None:
        if not base_index_savename.endswith(".index"):
            base_index_savename += ".index"
        faiss.write_index(self.base_index, base_index_savename)

    @property
    def path_to_model(self) -> os.path:
        if not self._path_to_model:
            print("Loading Universal Sentence Encoder from tfhub.dev...")
            self._path_to_model = "https://tfhub.dev/google" \
                                  "/universal-sentence-encoder/2"
        return self._path_to_model

    @path_to_model.setter
    def path_to_model(self, new_path) -> None:
        try:
            assert os.path.exists(new_path)
            self._path_to_model = new_path
        except AssertionError:
            print("{} is not a valid path".format(new_path))

    def load_model(self) -> None:
        self._model = hub.Module(self.path_to_model)

    @property
    def model(self) -> hub.Module:
        if not self._model:
            self.load_model()
        return self._model

    @property
    def is_sess_active(self) -> bool:
        return self._sess_active

    @property
    def sess(self) -> tf.Session:
        if not self._sess or not self.is_sess_active:
            if not self._model:
                self.load_model()
            self._sess = tf.Session()
            print("Initializing TF Session...")
            self._sess.run([tf.global_variables_initializer(),
                            tf.tables_initializer()])
            self._sess_active = True
            print("Session is active")
        return self._sess

    def close_sess(self) -> None:
        if self._sess:
            self.sess.close()
            print("Session closed")
        else:
            print("No active session")
        self._sess_active = False
        self._query_mode = False
        self._batch_mode = False

    @property
    def is_in_query_mode(self) -> bool:
        return self._query_mode

    def activate_query_mode(self) -> None:
        if not self._query_mode and not self.is_sess_active:
            print("Activating session in query mode...")
            self._query_mode = True
            self._batch_mode = False
        elif not self._query_mode and self.is_sess_active:
            print("Resetting session to query mode...")
            self.close_sess()
            self._query_mode = True
            self._batch_mode = False
        qm = tf.constant(self._query_mode, dtype=tf.bool)
        print("  Query mode: {}".format(self.sess.run(qm)))

    @property
    def is_in_batch_mode(self) -> bool:
        return not self._batch_mode

    def activate_batch_mode(self) -> None:
        if not self.is_in_batch_mode and not self.is_sess_active:
            print("Activating session in batch mode...")
            self._query_mode = False
            self._batch_mode = True
        elif not self.is_in_batch_mode and self.is_sess_active:
            print("Resetting session to batch mode...")
            self.close_sess()
            self._query_mode = False
            self._batch_mode = True
        bm = tf.constant(self.is_in_batch_mode, dtype=tf.bool)
        print("  Batch mode: {}".format(self.sess.run(bm)))

    def str_to_vector(self, query_str: str) -> np.array:
        if not self._query_mode:
            self.activate_query_mode()
        return self.sess.run(self.model([query_str]))

    def query_index(self,
                    query: str,
                    search_k: int = 5000
                    ) -> (np.array, np.array):

        query_vector = self.str_to_vector(query)
        return self.grand_index.search(query_vector, k=search_k)
