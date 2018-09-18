from .key_value_storage import KeyValueStorage
import json
import requests
from elasticsearch import Elasticsearch

query_str = """{
                "query": {
                    "ids": {
                        "values": []
                    }
                }
            }"""


class ESAdapter(KeyValueStorage):
    def __init__(self, es_endpoint='http://localhost:9200', logstash_file_path='/tmp/logstash_input.jl', http_auth=None):
        KeyValueStorage.__init__(self)

        self.es = Elasticsearch([es_endpoint], show_ssl_warnings=False, http_auth=http_auth,retry_on_timeout=True)
        self.es_endpoint = es_endpoint
        self.logstash_file = open(logstash_file_path, mode='a')

    def get_record(self, record_id, table_name):
        # table_name = index in this case
        if not isinstance(record_id, list):
            record_id = [record_id]
        query = json.loads(query_str)
        query['query']['ids']['values'] = record_id
        url = '{}/{}/_search'.format(self.es_endpoint, table_name)
        print('url={}'.format(url))
        sources = list()
        r = None
        try:
            r = requests.post(url, data=json.dumps(query))
        except Exception as e:
            print('Exception:{} occured while calling elasticsearch'.format(str(e)))

        print(r)
        if r and r.status_code == 200:
            for hit in r.json()['hits']['hits']:
                sources.append(hit['_source'])
        return sources

    def insert_record(self, record_id, record, table_name):
        for k in list(record):
            if ':' in k:
                record[k.split(':')[1]] = record[k]
                record.pop(k)
        record['faiss_id'] = record_id
        self.logstash_file.write('{}\n'.format(json.dumps(record)))

    def create_table(self, table_name):
        print('no way')

    def tables(self):
        print('thats not how it works')

    def insert_records_batch(self, records, table_name):
        for record in records:
            self.insert_record(record[0], record[1], table_name)

    def insert_record_es(self, record_id, record, doc_type, table_name):
        for k in list(record):
            if ':' in k:
                record[k.split(':')[1]] = record[k]
                record.pop(k)
        record['faiss_id'] = record_id
        self.write_to_es(table_name, doc_type, record)

    def write_to_es(self, table_name, doc_type, record):
        try:
            response = self.es.index(index=table_name,doc_type=doc_type , body=record)
            return response
        except Exception as e:
            print("Exception : {} occured while writing in elasticsearch".format(repr(e)))
            raise e