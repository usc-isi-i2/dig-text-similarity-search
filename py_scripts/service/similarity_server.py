# <editor-fold desc="Basic Imports">
from __future__ import unicode_literals
from flask import Flask, jsonify
from flask import request
from flask_cors import CORS

import os.path as p
import json
import traceback
from argparse import ArgumentParser

import sys
sys.path.append(p.join(p.dirname(__file__), '..'))
sys.path.append(p.join(p.dirname(__file__), '../..'))
# </editor-fold>

# <editor-fold desc="Parse Command Line Options">
arp = ArgumentParser(description='Deploy multiple faiss index shards '
                                 'as a RESTful API.')

arp.add_argument('index_dir_path', help='Path to index shards.')
arp.add_argument('-c', '--centroids', type=int, default=1,
                 help='Number of centroids to visit during search. '
                      '(Speed vs. Accuracy trade-off)')
arp.add_argument('-l', '--large', action='store_true',
                 help='Toggle large Universal Sentence Encoder (Transformer). '
                      'Note: Encoder and Faiss embedding spaces must match!')
arp.add_argument('-d', '--debug', action='store_true', default=False,
                 help='Increases verbosity of Flask app.')
arp.add_argument('-r', '--range_search', action='store_true', default=False,
                 help='Performs query-vector proximity search within an '
                      'L2 radius of 1.2 (instead of the default faiss k-search)')
opts = arp.parse_args()
# </editor-fold>


from dt_sim.processor.query_processor import QueryProcessor
from dt_sim.indexer.ivf_index_handlers import DeployShards, RangeShards
from dt_sim.vectorizer.sentence_vectorizer import DockerVectorizer

from py_scripts.configs.config import std_config, lrg_config


#### CONFIGURE ####
if opts.large:
    my_config = lrg_config
else:
    my_config = std_config

# Change index paths if necessary
if opts.index_dir_path:
    my_config['faiss_index_path'] = p.abspath(opts.index_dir_path)


#### INIT ####
app = Flask(__name__)
CORS(app, supports_credentials=True)

print(' * Initializing Faiss Indexes')
if opts.range_search:
    faiss_indexer = RangeShards(my_config['faiss_index_path'],
                                nprobe=opts.centroids)
else:
    faiss_indexer = DeployShards(my_config['faiss_index_path'],
                                 nprobe=opts.centroids)

print(' * Initializing Query Vectorizer')   # Emb space Bool toggle
query_vectorizer = DockerVectorizer(large=my_config['large_emb_space'])

print(' * Initializing Query Processor')
qp = QueryProcessor(index_handler=faiss_indexer, query_vectorizer=query_vectorizer)


#### APP DEF ####
@app.route('/')
def hello():
    return 'DIG Text Similarity Search\n'


@app.route('/search', methods=['GET'])
def text_similarity_search():
    query = request.args.get('query', None)
    k = request.args.get('k', 10)
    # rerank_by_doc = request.args.get('rerank_by_doc', 'false')
    # rerank_by_doc = str(rerank_by_doc).lower() == 'true'
    if not query:
        return jsonify({'message': 'The service is not able to process null requests'}), 400

    try:
        results = qp.query_corpus(query, int(k))
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        print(''.join(lines))
        return jsonify({'message': str(e)}), 500

    return json.dumps(results), 200


@app.route('/faiss', methods=['PUT'])
def add_shard():
    shard_path = p.abspath(request.args.get('path', None))
    if not p.exists(shard_path):
        return jsonify({'message': 'Path does not exist: {}'.format(shard_path)}), 404

    try:
        qp.add_shard(shard_path)
        return jsonify({'message': 'Successfully added shard to faiss index'}), 201
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        print(''.join(lines))
        return jsonify({'message': str(e)}), 500


#### MAIN ####
def main():
    app.run(host=my_config['host'], port=my_config['port'],
            threaded=True, debug=opts.debug)


if __name__ == '__main__':
    # TODO: Cache ifps on startup
    main()
