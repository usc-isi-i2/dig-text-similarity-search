from vectorizer.sentence_vectorizer import SentenceVectorizer
from indexer.faiss_indexer import FaissIndexer
from storage.es_adapter import ESAdapter
from process_documents.document_processor import DocumentProcessor

import json
import requests
from time import time
import pandas as pd
from optparse import OptionParser

es_url = 'http://dig:dIgDiG@mydig-sage-internal.isi.edu:80/es/'
es_index = 'sage_news'
table = 'dig-text-similarity-search'
faiss_path = '../data/testing/faiss_new_sample_index.index'

ids_query_str = """{
  "query": {
    "ids": {
      "values": []
    }
  }
}"""

if __name__ == '__main__':
    option_parser = OptionParser()
    option_parser.add_option('-q', '--query', dest='query')
    option_parser.add_option('-a', '--query_file', dest='query_file')
    option_parser.add_option('-o', '--output', dest='output', default='/tmp/evaluation.csv')
    option_parser.add_option('-f', '--faiss', dest='faiss', help='path to faiss index', default=faiss_path)
    option_parser.add_option('-t', '--table', dest='table', help='faiss id to sentence id mapping table', default=table)
    option_parser.add_option('-e', '--es', dest='es', default=es_url)
    option_parser.add_option('-i', '--index', dest='index', default=es_index)
    option_parser.add_option('-k', '--knearest', dest='k', default=10)

    (opts, args) = option_parser.parse_args()
    ifp = opts.query
    output = opts.output
    faiss = opts.faiss
    table = opts.table
    es_url = opts.es
    es_index = opts.index
    k = opts.k
    query_file = opts.query_file

    fi = FaissIndexer(faiss)
    sentence_vectorizer = SentenceVectorizer()
    es_adapter = ESAdapter()

    dp = DocumentProcessor(fi, sentence_vectorizer, es_adapter, table_name=table)

    if query_file:
        ifps = open(query_file).readlines()
    else:
        ifps = [ifp]

    evaluation_list = list()

    for ifp in ifps:
        start_time = time()
        results = dp.query_text(ifp, k=k)
        time_taken = time() - start_time

        doc_ids = [x['doc_id'] for x in results]
        ids_query = json.loads(ids_query_str)
        ids_query['query']['ids']['values'] = doc_ids

        es_response = requests.post('{}/_search'.format(es_url), json=ids_query)

        es_results = es_response.json()['hits']['hits']

        doc_dict = {}
        for hit in es_results:
            doc_dict[hit['_id']] = hit['_source']['knowledge_graph']['description'][0]['value']

        for result in results:
            doc_id = result['doc_id']
            if doc_id in doc_dict:
                evaluation_list.append(
                    (ifp, doc_id, doc_dict[doc_id], result['sentence_id'], result['sentence'], result['score'],
                     str(time_taken), -1))

    df = pd.DataFrame(data=evaluation_list,
                      columns=['ifp', 'doc_id', 'doc_text', 'sentence_id', 'sentence_text', 'score', 'query_time',
                               'relevance'])
    df.to_csv(output, index=False)
