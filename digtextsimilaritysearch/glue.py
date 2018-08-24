from vectorizer.sentence_vectorizer import SentenceVectorizer
from indexer.faiss_indexer import FaissIndexer
from storage.hbase_adapter import HBaseAdapter
from process_documents.document_processor import DocumentProcessor

import os
import json


cwd = os.getcwd()
doc_file = 'data/testing/news_sample.jl'
doc_file_path = os.path.join(cwd, doc_file)
docs = [json.loads(x) for x in open(doc_file_path)]

fi = FaissIndexer()
sentence_vectorizer = SentenceVectorizer()
hbase_adapter = HBaseAdapter('localhost')
dp = DocumentProcessor(fi, sentence_vectorizer, hbase_adapter, save_vectors=True)

dp.index_documents(docs, load_vectors=False)

print(dp.query_text("what is the moving annual return"))
