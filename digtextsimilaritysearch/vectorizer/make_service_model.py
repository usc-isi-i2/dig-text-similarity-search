import tensorflow as tf
import tensorflow_hub as hub

MODEL_NAME = 'USE-liteQuery-v1'
MODEL_LINK = 'https://tfhub.dev/google/universal-sentence-encoder/1'
VERSION = '001'
SERVE_PATH = './service/{}/{}'.format(MODEL_NAME, VERSION)

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
