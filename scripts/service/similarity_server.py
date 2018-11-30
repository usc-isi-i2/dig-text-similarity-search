# <editor-fold desc="Basic Imports">
from __future__ import unicode_literals
from flask import Flask, jsonify
from flask import request
from flask_cors import CORS

import os
import sys
import json
import traceback

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
# </editor-fold>

from dt_sim_api.processor.query_processor import QueryProcessor
from dt_sim_api.indexer.ivf_index_handlers import DeployShards, RangeShards
from dt_sim_api.vectorizer.sentence_vectorizer import DockerVectorizer

from scripts.configs.config import std_config, lrg_config

from optparse import OptionParser
options = OptionParser()
options.add_option('-i', '--index_dir_path', type='str')
options.add_option('-c', '--centroids', type='int', default=16)
options.add_option('-r', '--range_search', action='store_true', default=False)
options.add_option('-l', '--large', action='store_true', default=False)
options.add_option('-d', '--debug', action='store_true', default=False)
options.add_option('-A', '--AWS', action='store_true', default=False)   # Internal
(opts, _) = options.parse_args()


##### CONFIGURE #####
if opts.large:
    my_config = lrg_config
else:
    my_config = std_config

# Change index paths if necessary
if opts.index_dir_path:
    my_config['faiss_index_path'] = os.path.abspath(opts.index_dir_path)

# Internal
if opts.AWS:
    my_config = lrg_config
    my_config['faiss_index_path'] = '/faiss/news_day_shards/'


##### INIT #####
app = Flask(__name__)
CORS(app, supports_credentials=True)

print(' * Initializing Faiss Indexes')
if opts.range_search:
    faiss_indexer = RangeShards(my_config['faiss_index_path'], nprobe=opts.centroids)
else:
    faiss_indexer = DeployShards(my_config['faiss_index_path'], nprobe=opts.centroids)

print(' * Initializing Query Vectorizer')
query_vectorizer = DockerVectorizer(large=my_config['large_emb_space'])

print(' * Initializing Query Processor')
qp = QueryProcessor(index_handler=faiss_indexer, query_vectorizer=query_vectorizer)


##### APP DEF #####
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
    shard_path = os.path.abspath(request.args.get('path', None))
    if not os.path.exists(shard_path):
        return jsonify({'message': 'Path does not exist: {}'.format(shard_path)}), 404

    try:
        qp.add_shard(shard_path)
        return jsonify({'message': 'Successfully added shard to faiss index'}), 201
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        print(''.join(lines))
        return jsonify({'message': str(e)}), 500


##### MAIN #####
def main():
    app.run(host=my_config['host'], port=my_config['port'],
            threaded=True, debug=opts.debug)


if __name__ == '__main__':
    # TODO: Cache ifps on startup
    main()
