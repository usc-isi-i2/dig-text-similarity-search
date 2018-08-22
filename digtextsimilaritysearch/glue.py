from vectorizer.batch_vectorizer import BatchVectorizer
from indexer.faiss_indexer import FaissIndexer
from storage.hbase_adapter import HBaseAdapter
from process_documents.document_processor import DocumentProcessor

import json

docs = [json.loads(x) for x in open(
    '/Users/amandeep/Github/dig-text-similarity-search/digtextsimilaritysearch/unit_tests/resources/news_sample.jl')]
batch_vectorizer = BatchVectorizer()
fi = FaissIndexer()
hbase_adapter = HBaseAdapter('localhost')
dp = DocumentProcessor(fi, batch_vectorizer, None, hbase_adapter)
dp.index_documents(docs)
print(dp.query_text("what is the moving annual return"))