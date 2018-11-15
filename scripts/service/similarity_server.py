from __future__ import unicode_literals
from flask import Flask, jsonify
from flask import request
from flask_cors import CORS

import os
import json
import sys
import traceback

from indexer.IVF_disk_index_handler import DeployShards
from process_documents.document_processor import DocumentProcessor
from vectorizer.sentence_vectorizer import DockerVectorizer
from storage.es_adapter import ESAdapter
from config import config

app = Flask(__name__)
CORS(app, supports_credentials=True)

print('Initializing Batch Vectorizer')
query_vectorizer = DockerVectorizer()

print('Initializing Faiss Indexer')
faiss_indexer = DeployShards(config['faiss_index_path'])

print('Initializing ES Adapter')
es_adapter = ESAdapter(es_endpoint=config['es_endpoint'])

print('Initializing Document Processor')
dp = DocumentProcessor(indexer=faiss_indexer, vectorizer=query_vectorizer,
                       storage_adapter=None, table_name=config['es_index'])


@app.route("/")
def hello():
    return "DIG Text Similarity Search\n"


@app.route("/search", methods=['GET'])
def text_similarity_search():
    query = request.args.get("query", None)
    k = request.args.get("k", 10)
    rerank_by_doc = request.args.get("rerank_by_doc", 'false')
    rerank_by_doc = str(rerank_by_doc).lower() == 'true'
    if not query:
        return jsonify({"message": "The service is not able to process null request"}), 400

    try:
        results = dp.query_text(query, k=int(k), rerank_by_doc=rerank_by_doc)
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        print(''.join(lines))
        return jsonify({"message": str(e)}), 500

    return json.dumps(results), 200


@app.route("/faiss", methods=['PUT'])
def add_shard():
    shard_path = request.args.get("path", None)
    if not os.path.exists(shard_path):
        return jsonify({"message": "Path does not exist: {}".format(shard_path)}), 404

    try:
        dp.indexer.add_shard(shard_path)
        return jsonify({'message': 'successfully added shard to faiss index'}), 201
    except Exception as e:
        return jsonify({"message": str(e)}), 500


def main():
    app.run(host=config['host'], port=config['port'], threaded=True, debug=False)


if __name__ == '__main__':
    main()