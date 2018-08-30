from vectorizer.sentence_vectorizer import SentenceVectorizer
from indexer.faiss_indexer import FaissIndexer
from storage.hbase_adapter import HBaseAdapter
from storage.es_adapter import ESAdapter
from process_documents.document_processor import DocumentProcessor

import os
import json


cwd = os.getcwd()
doc_file = '../data/testing/news_sample.jl'
doc_file_path = os.path.join(cwd, doc_file)
docs = [json.loads(x) for x in open(doc_file_path)]

fi = FaissIndexer()
sentence_vectorizer = SentenceVectorizer()
# hbase_adapter = HBaseAdapter('localhost')
es_adapter = ESAdapter()
dp = DocumentProcessor(fi, sentence_vectorizer, es_adapter, save_vectors=True, logstash_input_file='/tmp/test.jl', table_name='dig-text-similarity-search')

dp.index_documents(docs, load_vectors=False)

print(dp.query_text("what is the moving annual return"))
