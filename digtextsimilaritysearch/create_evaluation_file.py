from vectorizer.sentence_vectorizer import SentenceVectorizer
from indexer.faiss_indexer import FaissIndexer
from storage.es_adapter import ESAdapter
from process_documents.document_processor import DocumentProcessor
from config import config

import os
import json
import requests
from time import time
import pandas as pd

es_url = 'http://dig:dIgDiG@mydig-sage-internal.isi.edu:80/es/'
es_index = 'sage_news'

ids_query_str = """{
  "query": {
    "ids": {
      "values": []
    }
  }
}"""

cwd = os.getcwd()
doc_file = '../data/testing/news_sample.jl'
doc_file_path = os.path.join(cwd, doc_file)
docs = [json.loads(x) for x in open(doc_file_path)]

fi = FaissIndexer(path_to_index_file=config['faiss_index_path'])
sentence_vectorizer = SentenceVectorizer()
es_adapter = ESAdapter()

dp = DocumentProcessor(fi, sentence_vectorizer, es_adapter, table_name='dig-text-similarity-search')

ifp = 'what is the moving annual return'

start_time = time()
results = dp.query_text(ifp)
time_taken = time() - start_time

doc_ids = [x['doc_id'] for x in results]
ids_query = json.loads(ids_query_str)
ids_query['query']['ids']['values'] = doc_ids

es_response = requests.post('{}/_search'.format(es_url), json=ids_query)

es_results = es_response.json()['hits']['hits']

doc_dict = {}
for hit in es_results:
    doc_dict[hit['_id']] = hit['_source']['knowledge_graph']['description'][0]['value']

evaluation_list = list()

for result in results:
    doc_id = result['doc_id']
    evaluation_list.append((ifp, doc_id, doc_dict[doc_id], result['sentence_id'], result['sentence'], result['score'],
                            str(time_taken), -1))

df = pd.DataFrame(data=evaluation_list,
                  columns=['ifp', 'doc_id', 'doc_text', 'sentence_id', 'sentence_text', 'score', 'query_time',
                           'relevance'])
df.to_csv('../data/evaluation_file.csv', index=False)
# print(json.dumps(evaluation_list, indent=2))
