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
        print("Initializing TF Session...")
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
