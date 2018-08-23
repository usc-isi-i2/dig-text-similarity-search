from __future__ import unicode_literals
from flask import Flask, jsonify
from flask import request
from flask_cors import CORS, cross_origin
import json

from indexer.faiss_indexer import FaissIndexer
from process_documents.document_processor import DocumentProcessor
from vectorizer.batch_vectorizer import BatchVectorizer
from storage.hbase_adapter import HBaseAdapter
from config import config

app = Flask(__name__)
CORS(app, supports_credentials=True)

query_vectorizer = None
faiss_indexer = None
hbase_adapter = None
dp = None


def initialize():
    global query_vectorizer
    global faiss_indexer
    global hbase_adapter
    global dp
    if not query_vectorizer:
        print('Initializing Batch Vectorizer')
        query_vectorizer = BatchVectorizer()
    if not faiss_indexer:
        print('Initializing Faiss Indexer')
        faiss_indexer = FaissIndexer(path_to_index_file=config['faiss_index_path'])
    if not hbase_adapter:
        print('Initializing Hbase Adapter')
        hbase_adapter = HBaseAdapter(config['hbase_server'])
    if not dp:
        print('Initializing Document Processor')
        dp = DocumentProcessor(faiss_indexer, query_vectorizer, hbase_adapter)


@app.route("/")
def hello():
    return "DIG Text Similarity Search\n"


@app.route("/search", methods=['GET'])
def text_similarity_search():
    initialize()
    global dp
    query = request.args.get("query", None)
    k = request.args.get("k", 3)
    if not query:
        return jsonify({"message": "The service is not able to process null request"}), 400
    try:
        results = dp.query_text(query,k=k)
    except Exception as e:
        return jsonify({"message": str(e)}), 500

    return json.dumps(results), 200


if __name__ == '__main__':
    # initialize()
    app.run(host=config['host'], port=config['port'], threaded=False, debug=False)
