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
        config = tf.ConfigProto(device_count={'CPU': 8},
                                intra_op_parallelism_threads=8,
                                inter_op_parallelism_threads=8)

        self.session = tf.Session(config=config)

        print('Initializing TF Session...')
        with self.graph.as_default():
            self.session.run([tf.global_variables_initializer(), tf.tables_initializer()])

    def close_session(self):
        print('Closing TF Session...')
        self.session.close()

    def make_vectors(self, sentences, batch_size=512) -> List[tf.Tensor]:
        embeddings = list()
        batched_tensors = list()
        if not isinstance(sentences, list):
            sentences = [sentences]
        with self.graph.as_default():
            if len(sentences) > batch_size:
                while len(sentences) >= batch_size:
                    batch, sentences = list(sentences[:batch_size]), list(sentences[batch_size:])
                    batched_tensors.append(tf.constant(batch, dtype=tf.string))

                dataset = tf.data.Dataset.from_tensor_slices(batched_tensors)
                dataset = dataset.make_one_shot_iterator()

                make_embeddings = self.model(dataset.get_next())

                while True:
                    try:
                        embeddings.append(self.session.run(make_embeddings))
                    except tf.errors.OutOfRangeError:
                        break

            if len(sentences):
                basic_batch = self.model(sentences)
                embeddings.append(self.session.run(basic_batch))

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

    # TODO: Depreciate save_vectors
    @staticmethod
    def save_with_ids(file_path, embeddings, sentences, sent_ids):
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.vstack(embeddings).astype(np.float32)
        if not isinstance(sentences, np.ndarray):
            sentences = np.array(sentences, dtype=np.str)
        if not isinstance(sent_ids, np.ndarray):
            try:
                sent_ids = np.array(sent_ids, dtype=np.int64)
            except ValueError:
                print(sent_ids)

        np.savez_compressed(file=file_path, sent_ids=sent_ids,
                            embeddings=embeddings, sentences=sentences)

    # TODO: Depreciate load_vectors
    @staticmethod
    def load_with_ids(file_path):
        if not file_path.endswith('.npz'):
            file_path += '.npz'

        loaded = np.load(file=file_path, mmap_mode='r')
        embeddings = loaded['embeddings']
        sentences = loaded['sentences']
        sent_ids = loaded['sent_ids']
        return embeddings, sentences, sent_ids
