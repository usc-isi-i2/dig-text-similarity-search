import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from typing import List


class SentenceVectorizer(object):

    def __init__(self, path_to_model=None):

        model_dir = os.path.join(os.path.dirname(__file__), 'model/')
        model_loc = '1fb57c3ffe1a38479233ee9853ddd7a8ac8a8c47'
        model_loc = os.path.join(model_dir, model_loc)
        if not path_to_model and os.path.isdir(model_loc):
            path_to_model = model_loc
        elif not path_to_model and os.path.isdir(model_dir):
            os.environ['TFHUB_CACHE_DIR'] = model_dir

        if not path_to_model:
            path_to_model = 'https://tfhub.dev/google/universal-sentence-encoder/2'

        self.graph = tf.get_default_graph()
        print('Loading model: {}'.format(path_to_model))
        with self.graph.as_default():
            self.model = hub.Module(path_to_model)
        print('Done loading model')

        self.session = None

        self.start_session()

    def start_session(self):
        self.session = tf.Session()

        print('Initializing TF Session...')
        with self.graph.as_default():
            self.session.run([tf.global_variables_initializer(), tf.tables_initializer()])

    def close_session(self):
        print('Closing TF Session...')
        self.session.close()

    def make_vectors(self, sentences) -> List[tf.Tensor]:
        with self.graph.as_default():
            batched_tensors = list()
            batched_tensors.append(tf.constant(sentences, dtype=tf.string))

            dataset = tf.data.Dataset.from_tensor_slices(batched_tensors).make_one_shot_iterator()

            make_embeddings = self.model(dataset.get_next())

            embeddings = self.session.run(make_embeddings)

        return embeddings

    @staticmethod
    def save_vectors(embeddings: List[tf.Tensor], sentences: List[object], file_path):
        """
        Converts embedding tensors and corresponding list of sentences into np.arrays,
        then saves both arrays in the same compressed .npz file

        Note: .npz will be appended to file_path automatically

        :param embeddings: List of tensors made from sentences
        :param sentences: List of corresponding sentences
        :param file_path: /full/path/to/file_name
        :return: Writes zipped archive to disk
        """

        embeddings = np.array(embeddings, dtype=np.float32)
        sentences = np.array(sentences, dtype=object)

        np.savez_compressed(file=file_path, embeddings=embeddings, sentences=sentences)

    @staticmethod
    def load_vectors(file_path, mmap=True):
        """
        Loads zipped archive containing embeddings and sentences as two separate np.arrays

        :param file_path: /full/path/to/file_name.npz
        :param mmap: If mmap is True, the .npz file is not read from disk, not into memory
        :return: embeddings and sentences as separate np.arrays
        """

        if not file_path.endswith('.npz'):
            file_path += '.npz'

        if mmap:
            mode = 'r'
        else:
            mode = None

        loaded = np.load(file=file_path, mmap_mode=mode)
        embeddings = loaded['embeddings']
        sentences = loaded['sentences']

        return embeddings, sentences
