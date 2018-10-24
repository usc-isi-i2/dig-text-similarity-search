import os
import tensorflow as tf
import tensorflow_hub as hub
from optparse import OptionParser
# <editor-fold desc="Parse Command Line Options">
options = OptionParser()
options.add_option('-v', '--version', default='001')
(opts, _) = options.parse_args()
# </editor-fold>

"""
Makes file to run Universal Sentence Encoder with docker 
    for efficient vectorization

First run this script with:
    $ ./prep_service_model.sh
    OR
    $ source activate dig_text_similarity 
    $ python make_service_model.py -v {version int}

Then run docker with:
    $ ./run_service_model.sh
    OR 
    $ docker pull tensorflow/serving
    $ docker run -p 8501:8501 \ 
        --mount type=bind,source={/path/to}/USE-lite-v2/,target=/models/USE-lite-v2 \ 
        -e MODEL_NAME=USE-lite-v2 -t tensorflow/serving

Options:
    -v Version number of model (highest version number will be deployed)
"""

# Find model
model_dir = 'model/1fb57c3ffe1a38479233ee9853ddd7a8ac8a8c47'
model_dir = os.path.join(os.path.dirname(__file__), model_dir)
if os.path.isdir(model_dir):
    model_link = model_dir
else:
    model_link = 'https://tfhub.dev/google/universal-sentence-encoder/2'

# Model type and version
MODEL_LINK = model_link
MODEL_NAME = 'USE-lite-v2'
VERSION = opts.version
SERVE_PATH = os.path.join(os.path.dirname(__file__),
                          './service_models/{}/{}'.format(MODEL_NAME, VERSION))

# Build graph
with tf.Graph().as_default():
    module = hub.Module(MODEL_LINK, name=MODEL_NAME)
    text = tf.placeholder(tf.string, shape=[None], name='text')
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
