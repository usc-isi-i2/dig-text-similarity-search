import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from typing import List


class BatchVectorizer(object):

    def __init__(self, path_to_model='https://tfhub.dev/google/universal-sentence-encoder/2', batch_mode=True):
        print('Loading model: {}'.format(path_to_model))
        self.model = hub.Module(path_to_model)
        print('Done loading model')
        self.batch_mode = tf.constant(batch_mode, dtype=tf.bool)

        self.session = tf.Session()
        print('Initializing TF Session...')
        self.session.run([tf.global_variables_initializer(),
                          tf.tables_initializer()])

    def make_vectors(self, sentences) -> List[tf.Tensor]:

        batched_tensors = list()
        batched_tensors.append(tf.constant(sentences, dtype=tf.string))

        dataset = tf.data.Dataset.from_tensor_slices(batched_tensors).make_one_shot_iterator()

        make_embeddings = self.model(dataset.get_next())
        embeddings = list()

        try:
            embeddings.append(self.session.run(make_embeddings))

        except tf.errors.OutOfRangeError:
            self.session.close()
            pass

        return embeddings

    @staticmethod
    def save_vectors(embeddings: List[tf.Tensor],
                     sentences: List[str],
                     file_path):
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
    def load_vectors(file_path):
        """
        Loads zipped archive containing embeddings and sentences as two separate np.arrays

        :param file_path: /full/path/to/file_name.npz
        :return: embeddings and sentences as separate np.arrays
        """

        if not file_path.endswith('.npz'):
            file_path += '.npz'

        loaded = np.load(file=file_path)
        embeddings = loaded['embeddings']
        sentences = loaded['sentences']

        return embeddings, sentences
