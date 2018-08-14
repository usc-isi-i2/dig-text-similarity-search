import os

import numpy as np
import tensorflow as tf

from typing import List
from base_index_handler import Base


class Collector(Base):

    def __init__(self,
                 path_to_index_dir: str,
                 base_index_name: str,
                 grand_index_name: str,
                 path_to_model: str = None
                 ) -> None:

        self._path_to_index_dir = path_to_index_dir

        if not grand_index_name.endswith(".index"):
            grand_index_name += ".index"

        if not base_index_name.endswith(".index"):
            base_index_name += ".index"

        Base.__init__(self,
                      index_dir=path_to_index_dir,
                      grand_index_name=grand_index_name,
                      base_index_name=base_index_name,
                      path_to_model=path_to_model)

        # TODO: Make vector index map a database on disk
        self._index_vector_map = None
        # TODO: Same with sent_dicts
        self._sent_dicts = None

    @property
    def path_to_index_dir(self) -> os.path:
        if not os.path.isdir(self._path_to_index_dir):
            os.mkdir(self._path_to_index_dir)
        return self._path_to_index_dir

    @property
    def index_vector_map(self) -> dict:
        if not self._index_vector_map:
            self._index_vector_map = dict()
        return self._index_vector_map

    def add_to_index_map(self,
                         new_addition_map: dict
                         ) -> None:

        for sent_id, idx_row in new_addition_map.items():
            try:
                assert idx_row not in self.index_vector_map, \
                    "Duplicate vector detected"
                self.index_vector_map[idx_row] = sent_id
            except AssertionError:
                print("Row {} is already occupied by {}".format(sent_id,
                                        self.index_vector_map[idx_row]))

    @property
    def sentence_dicts(self) -> List[dict]:
        if not self._sent_dicts:
            self._sent_dicts = list()
        return self._sent_dicts

    def extend_sentence_dicts(self,
                              new_sent_dicts: List[dict]
                              ) -> None:

        self.sentence_dicts.extend(new_sent_dicts)

    @staticmethod
    def initialize_sent_dicts(doc_list: List[dict],
                              id_joiner: str = "|::|"
                              ) -> List[dict]:

        sent_dict_batch = list()
        for doc in doc_list:
            for j, sent in enumerate(doc["lexisnexis"]["split_doc_description"]):
                sent_dict = dict()

                sent_dict["doc_id"] = doc["doc_id"]
                sent_dict["text"] = sent
                sent_dict["tag"] = id_joiner.join([doc["doc_id"], str(j)])
                sent_dict["emb"] = None
                sent_dict_batch.append(sent_dict)

        return sent_dict_batch

    @staticmethod
    def batch_to_dataset(sent_dict_batch: List[dict],
                         batch_size: int = 50
                         ) -> tf.data.Iterator:

        batched_tensors = list()

        minibatch = list()
        for sent_dict in sent_dict_batch:
            minibatch.append(sent_dict["text"])

            if len(minibatch) == batch_size:
                minibatch_tensor = tf.constant(minibatch, dtype=tf.string)
                batched_tensors.append(minibatch_tensor)
                minibatch = list()

        while len(minibatch) < batch_size:
            minibatch.append("")
        batched_tensors.append(minibatch)

        dataset = tf.data.Dataset.from_tensor_slices(batched_tensors)
        return dataset.make_one_shot_iterator()

    def make_vectors(self,
                     dataset: tf.data.Iterator
                     ) -> List[tf.Tensor]:

        make_embeddings = self.model(dataset.get_next())
        embeddings = list()
        while True:
            try:
                embeddings.append(self.sess.run(make_embeddings))
            except tf.errors.OutOfRangeError:
                self.close_sess()
                break

        return embeddings

    @staticmethod
    def tensors_to_list(embedding_tensors: List[tf.Tensor],
                        partial_sent_dicts: List[dict]
                        ) -> List[dict]:

        complete_sent_dicts = list()
        for minibatch in embedding_tensors:
            batch = list(np.array(minibatch).tolist())

            for vector in batch:
                if len(partial_sent_dicts) == 0:
                    break
                sent_dict = partial_sent_dicts.pop(0)
                sent_dict["emb"] = vector
                complete_sent_dicts.append(sent_dict)

        return complete_sent_dicts

    @staticmethod
    def format_vectors(complete_sent_dicts: List[dict],
                       offset: int = 0
                       ) -> (np.array, dict):

        vector_list = list()
        index_vector_map = dict()
        for i, sent_dict in enumerate(complete_sent_dicts):
            vector_list.append(sent_dict["emb"])
            index_vector_map[sent_dict["tag"]] = i + offset

        vector_array = np.array(vector_list).astype("float32")
        return vector_array, index_vector_map

    def add_to_index(self,
                     docs_to_add: List[dict],
                     start_from_base: bool = False
                     ) -> None:

        if start_from_base:
            self.set_grand_index(self.base_index)

        init_sent_dicts = self.initialize_sent_dicts(docs_to_add)
        tensor_str_iter = self.batch_to_dataset(init_sent_dicts)

        self.activate_batch_mode()
        vector_tensors = self.make_vectors(tensor_str_iter)

        sent_dicts = self.tensors_to_list(embedding_tensors=vector_tensors,
                                          partial_sent_dicts=init_sent_dicts)
        self.extend_sentence_dicts(sent_dicts)

        vectors, idx_vec_map = self.format_vectors(complete_sent_dicts=sent_dicts,
                                                   offset=self.grand_index.ntotal)

        self.add_to_index_map(idx_vec_map)
        self.grand_index.add(vectors)

    def query_index(self,
                    query: str,
                    search_k: int = 5000
                    ) -> List[dict]:

        query_vector = self.str_to_vector(query)
        Sim, Idx = self.grand_index.search(query_vector, k=search_k)

        return self.refine_results(similarity_scores=Sim, index_indices=Idx)

    def refine_results(self,
                       similarity_scores: np.array,
                       index_indices: np.array,
                       id_joiner: str = "|::|"
                       ) -> List[dict]:

        intermediate_results = list()
        for Sim, Idx in zip(list(similarity_scores.tolist())[0],
                            list(index_indices.tolist())[0]):
            try:
                tag = self.index_vector_map[Idx]
                each_result = dict()
                each_result["sim"] = Sim
                each_result["tag"] = tag
                each_result["doc_id"] = tag.split(id_joiner)[0]
                each_result["sent_num"] = tag.split(id_joiner)[1]
                for sent_dict in self.sentence_dicts:
                    if each_result["tag"] == sent_dict["tag"]:
                        each_result["text"] = sent_dict["text"]
                        intermediate_results.append(each_result)
            except KeyError:
                continue

        final_docs = list()
        for ranked_result in intermediate_results:
            ranked_doc = dict()
            ranked_doc["doc_id"] = ranked_result["doc_id"]
            ranked_doc["sentences"] = list()
            if not any(ranked_doc["doc_id"] in doc["doc_id"]
                       for doc in final_docs):
                final_docs.append(ranked_doc)

        for ranked_doc in final_docs:
            for ranked_result in intermediate_results:
                this_result = dict()
                if ranked_doc["doc_id"] == ranked_result["doc_id"]:
                    this_result["sim"] = ranked_result["sim"]
                    this_result["text"] = ranked_result["text"]
                    this_result["sent_num"] = ranked_result["sent_num"]
                if len(this_result) > 0:
                    ranked_doc["sentences"].append(this_result)

        return list(final_docs)