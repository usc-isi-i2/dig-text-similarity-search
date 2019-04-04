# <editor-fold desc="Basic Imports">
from __future__ import unicode_literals
from flask import Flask, jsonify
from flask import request
from flask_cors import CORS

import os.path as p
import json
import traceback
from datetime import date, timedelta
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
                      'Speed vs. Accuracy trade-off. (Default = 1)')
arp.add_argument('-r', '--radius', type=float, default=0.65,
                 help='Specify the maximum L2 distance from the query vector. '
                      '(Default = 0.65, determined empirically)')
arp.add_argument('-l', '--large', action='store_true',
                 help='Toggle large Universal Sentence Encoder (Transformer). '
                      'Note: Encoder and Faiss embedding spaces must match!')
arp.add_argument('-a', '--also_load_nested', action='store_true', default=False,
                 help='Load indexes nested in sub directories of index_dir_path. ')
arp.add_argument('-d', '--debug', action='store_true', default=False,
                 help='Increases verbosity of Flask app.')
opts = arp.parse_args()
# </editor-fold>


from dt_sim.processor.query_processor import QueryProcessor
from dt_sim.indexer.ivf_index_handlers import RangeShards
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
faiss_indexer = RangeShards(my_config['faiss_index_path'],
                            nprobe=opts.centroids,
                            get_nested=opts.also_load_nested)

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
    k = int(request.args.get('k', 10))
    if not query:
        return jsonify({'message': 'The service is not able to process null requests'}), 400

    # Default date-range search: past 60 days
    end_date = request.args.get('end_date', date.isoformat(date.today()))
    if end_date > date.isoformat(date.today()):     # Handles erroneous future dates
        end_date = date.isoformat(date.today())
    end_dt_obj = date(*tuple(int(ymd) for ymd in end_date.split('-')))
    start_date = request.args.get('start_date', date.isoformat(end_dt_obj - timedelta(60)))
    if not start_date <= end_date:
        return jsonify({'message': 'Start-date must occur before end-date'}), 400

    # Max date-range: 180 day-span
    max_range = date.isoformat(end_dt_obj - timedelta(180))
    if max_range > start_date:
        start_date = max_range

    # Specify payload format
    rerank_by_doc = request.args.get('rerank_by_doc', 'false')
    rerank_by_doc = str(rerank_by_doc).lower() == 'true'

    try:
        results = qp.query_corpus(query, k=k, radius=opts.radius,
                                  start=start_date, end=end_date,
                                  rerank_by_doc=rerank_by_doc)
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
