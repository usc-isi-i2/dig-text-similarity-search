import os
import tensorflow as tf
import tensorflow_hub as hub

model_dir = os.path.join(os.path.dirname(__file__),
                         'model/1fb57c3ffe1a38479233ee9853ddd7a8ac8a8c47')
if os.path.isdir(model_dir):
    model_link = model_dir
else:
    model_link = 'https://tfhub.dev/google/universal-sentence-encoder/2'
MODEL_LINK = model_link
MODEL_NAME = 'USE-liteQuery-v2'
VERSION = '001'
SERVE_PATH = './service_models/{}/{}'.format(MODEL_NAME, VERSION)

with tf.Graph().as_default():
    module = hub.Module(MODEL_LINK, name=MODEL_NAME)
    text = tf.placeholder(tf.string, shape=(1,), name='text')
    embedding = module(text)

    init_op = tf.group([tf.global_variables_initializer(),
                        tf.tables_initializer()])

    with tf.Session() as sess:
        sess.run(init_op)
        tf.saved_model.simple_save(
            sess,
            SERVE_PATH,
            inputs={'text': text},
            outputs={'embedding': embedding},
            legacy_init_op=tf.tables_initializer()
        )
