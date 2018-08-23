from vectorizer.batch_vectorizer import BatchVectorizer
from vectorizer.query_vectorizer import QueryVectorizer
from indexer.faiss_indexer import FaissIndexer
from storage.hbase_adapter import HBaseAdapter
from process_documents.document_processor import DocumentProcessor

import json

docs = [json.loads(x) for x in open(
    '/Users/amandeep/Github/dig-text-similarity-search/digtextsimilaritysearch/unit_tests/resources/news_sample.jl')]
# docs = [json.loads(x) for x in open(
#     '/Users/amandeep/Github/dig-text-similarity-search/digtextsimilaritysearch/unit_tests/resources/new_2018-08-08.jl')]
batch_vectorizer = QueryVectorizer()
fi = FaissIndexer()
hbase_adapter = HBaseAdapter('localhost')
dp = DocumentProcessor(fi, batch_vectorizer, hbase_adapter, save_vectors=True)
dp.index_documents(docs, load_vectors=False)
print(dp.query_text("what is the moving annual return"))