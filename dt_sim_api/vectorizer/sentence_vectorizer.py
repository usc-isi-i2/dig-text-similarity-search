import os
import json
import requests
from time import time
from typing import List, Union

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from .base_vectorizer import BaseVectorizer


##### Query Vectorization #####
class DockerVectorizer(BaseVectorizer):
    """
    Intended for fast Query Vectorization
    """
    def __init__(self, url=None):
        BaseVectorizer.__init__(self)
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


##### Corpus Vectorization #####
class SentenceVectorizer(BaseVectorizer):
    """
    Intended for batch Corpus Vectorization
    """
    def __init__(self, path_to_model: str = None, large: bool = False,
                 intra: int = 8, inter: int = 2):
        BaseVectorizer.__init__(self)

        model_parent_dir = os.path.join(os.path.dirname(__file__), 'model/')
        if large:
            model_dir = '96e8f1d3d4d90ce86b2db128249eb8143a91db73/'
            model_url = 'https://tfhub.dev/google/universal-sentence-encoder-large/3'
            self.large_USE = True
        else:
            model_dir = '1fb57c3ffe1a38479233ee9853ddd7a8ac8a8c47/'
            model_url = 'https://tfhub.dev/google/universal-sentence-encoder/2'
        model_path = os.path.join(model_parent_dir, model_dir)

        self.path_to_model = path_to_model
        if not self.path_to_model and os.path.isdir(model_path):
            self.path_to_model = model_path
        elif not self.path_to_model and os.path.isdir(model_parent_dir):
            os.environ['TFHUB_CACHE_DIR'] = model_parent_dir
        if not self.path_to_model:
            self.path_to_model = model_url

        # TODO: fix TF Config
        self.config = tf.ConfigProto(device_count={'CPU': intra},
                                     allow_soft_placement=True,
                                     intra_op_parallelism_threads=intra,
                                     inter_op_parallelism_threads=inter)

        self.graph = None
        self.model = None
        print('Loading model: {}'.format(self.path_to_model))
        self.define_graph()
        print('Done loading model')
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

    def make_vectors(self, sentences: Union[str, List[str]],
                     n_minibatch: int = 512, verbose: bool = False) -> List[tf.Tensor]:
        if not isinstance(sentences, list):
            sentences = [sentences]
        i = 0
        t_st = time()
        timing = list()

        embeddings = list()
        batched_tensors = list()
        with self.graph.as_default():
            # High throughput vectorization (fast)
            if len(sentences) > n_minibatch:
                while len(sentences) >= n_minibatch:
                    batch, sentences = list(sentences[:n_minibatch]), list(sentences[n_minibatch:])
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
                            print('  ** {:5d}/{}'
                                  ' : {:3.3f}s :: {:3.3f}s avg'
                                  ''.format(i, len(batched_tensors),
                                            timing[-1], sum(timing)/len(timing)))
                            i += 1
                    except tf.errors.OutOfRangeError:
                        break

            # Tail end vectorization (slow)
            if len(sentences):
                t_1 = time()
                basic_batch = self.model(sentences)
                embeddings.append(self.session.run(basic_batch))
                if verbose:
                    tm, ts = divmod(time() - t_st, 60)
                    print('  ** {:5d}/{}'
                          ' : {:3.3f}s :: {}m{:.1f}s tot'
                          ''.format(i, len(batched_tensors),
                                    time() - t_1, int(tm), ts))
        return embeddings
