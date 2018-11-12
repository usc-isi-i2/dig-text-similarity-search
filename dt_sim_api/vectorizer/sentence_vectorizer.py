import os
import json
import requests
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from typing import List
from time import time


class DockerVectorizer(object):

    def __init__(self, url=None):
        if not url:
            url = 'http://localhost:8501/v1/models/USE-lite-v2:predict'
        self.url = url

    def make_vectors(self, query):
        if not isinstance(query, list):
            query = [query]

        payload = {"inputs": {"text": query}}
        payload = json.dumps(payload)

        response = requests.post(self.url, data=payload)
        response.raise_for_status()

        return response.json()['outputs']


class SentenceVectorizer(object):

    def __init__(self, path_to_model=None, intra=None, inter=None):

        model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'model/'))
        model_loc = '1fb57c3ffe1a38479233ee9853ddd7a8ac8a8c47'
        model_loc = os.path.join(model_dir, model_loc)
        self.path_to_model = path_to_model
        if not self.path_to_model and os.path.isdir(model_loc):
            self.path_to_model = model_loc
        elif not self.path_to_model and os.path.isdir(model_dir):
            os.environ['TFHUB_CACHE_DIR'] = model_dir

        if not self.path_to_model:
            self.path_to_model = 'https://tfhub.dev/google/universal-sentence-encoder/2'

        self.graph = None
        self.model = None
        print('Loading model: {}'.format(self.path_to_model))
        self.define_graph()
        print('Done loading model')

        if not intra:
            intra = 8
        if not inter:
            inter = 2
        self.config = tf.ConfigProto(device_count={'CPU': intra},
                                     allow_soft_placement=True,
                                     intra_op_parallelism_threads=intra,
                                     inter_op_parallelism_threads=inter)

        self.session = None
        print('Initializing TF Session...')
        self.start_session()

    def define_graph(self):
        self.graph = tf.get_default_graph()
        with self.graph.as_default():
            self.model = hub.Module(self.path_to_model)

    def start_session(self):
        self.session = tf.Session(config=self.config)
        with self.graph.as_default():
            self.session.run([tf.global_variables_initializer(), tf.tables_initializer()])

    def close_session(self):
        self.session.close()
        tf.reset_default_graph()
        self.define_graph()

    def make_vectors(self, sentences, batch_size=512, verbose=False) -> List[tf.Tensor]:
        i = 0
        t_st = time()
        timing = list()
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
                        t_0 = time()
                        embeddings.append(self.session.run(make_embeddings))
                        if verbose:
                            timing.append(time() - t_0)
                            print('  ** {:5d}/{} : {:3.3f}s :: {:3.3f}s avg'
                                  ''.format(i, len(batched_tensors),
                                            timing[-1], sum(timing)/len(timing)))
                            i += 1
                    except tf.errors.OutOfRangeError:
                        break

            if len(sentences):
                t_1 = time()
                basic_batch = self.model(sentences)
                embeddings.append(self.session.run(basic_batch))
                if verbose:
                    tm, ts = divmod(time() - t_st, 60)
                    print('  ** {:5d}/{} : {:3.3f}s :: {}m{:.1f}s tot'
                          ''.format(i, len(batched_tensors),
                                    time() - t_1, int(tm), ts))

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
    def save_with_ids(file_path, embeddings, sentences, sent_ids, compressed=True):
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.vstack(embeddings).astype(np.float32)
        if not isinstance(sentences, np.ndarray):
            sentences = np.array(sentences, dtype=np.str)
        if not isinstance(sent_ids, np.ndarray):
            try:
                sent_ids = np.array(sent_ids, dtype=np.int64)
            except ValueError:
                print(sent_ids)

        if compressed:
            np.savez_compressed(file=file_path, sent_ids=sent_ids,
                                embeddings=embeddings, sentences=sentences)
        else:
            np.savez(file=file_path, sent_ids=sent_ids,
                     embeddings=embeddings, sentences=sentences)

    # TODO: Depreciate load_vectors
    @staticmethod
    def load_with_ids(file_path, mmap=True):
        if not file_path.endswith('.npz'):
            file_path += '.npz'

        if mmap:
            mode = 'r'
        else:
            mode = None

        loaded = np.load(file=file_path, mmap_mode=mode)
        embeddings = loaded['embeddings']
        sentences = loaded['sentences']
        sent_ids = loaded['sent_ids']
        return embeddings, sentences, sent_ids