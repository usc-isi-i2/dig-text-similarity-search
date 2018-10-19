import os
import tensorflow as tf
import tensorflow_hub as hub
from optparse import OptionParser
# <editor-fold desc="Parse Command Line Options">
# Model type and version
options = OptionParser()
options.add_option('-b', '--batch', action='store_true', default=False)
options.add_option('-q', '--query', action='store_true', default=False)
options.add_option('-v', '--version', default='001')
(opts, _) = options.parse_args()
# </editor-fold>

"""
Makes file to run Universal Sentence Encoder with docker 
    for efficient vectorization

To run docker:
    $ docker pull tensorflow/serving
    $ docker run -p 8501:8501 \ 
        --mount type=bind,source={/path/to/model_name/},target=/models/{model_name} \ 
        -e MODEL_NAME={model_name} -t tensorflow/serving

    ( Replace {model_name} with USE-liteBatch-v2 or USE-liteQuery-v2 )

Options:
    -b Flag to make Batch model for preprocessing
    -q Flag to make Query model for online service
    -v Version number of model (highest version number will be deployed)
"""

# Find model
model_dir = 'model/1fb57c3ffe1a38479233ee9853ddd7a8ac8a8c47'
model_dir = os.path.join(os.path.dirname(__file__), model_dir)
if os.path.isdir(model_dir):
    model_link = model_dir
else:
    model_link = 'https://tfhub.dev/google/universal-sentence-encoder/2'

# Specify model name and input shape
model_name = None
model_shape = None
if opts.batch:
    model_name = 'USE-liteBatch-v2'
    model_shape = [None]
elif opts.query:
    model_name = 'USE-liteQuery-v2'
    model_shape = (1,)
assert not model_name, 'Please specify model type: Batch (-b) or Query (-q)'

# Model specifics
MODEL_LINK = model_link
MODEL_NAME = model_name
VERSION = opts.version
SERVE_PATH = './service_models/{}/{}'.format(MODEL_NAME, VERSION)

# Build graph
with tf.Graph().as_default():
    module = hub.Module(MODEL_LINK, name=MODEL_NAME)
    text = tf.placeholder(tf.string, shape=model_shape, name='text')
    embedding = module(text)

    init_op = tf.group([tf.global_variables_initializer(),
                        tf.tables_initializer()])

    # Save model for serving
    with tf.Session() as sess:
        sess.run(init_op)
        tf.saved_model.simple_save(
            sess,
            SERVE_PATH,
            inputs={'text': text},
            outputs={'embedding': embedding},
            legacy_init_op=tf.tables_initializer()
        )
