from .key_value_storage import KeyValueStorage
import json
import requests

query_str = """{
                "query": {
                    "ids": {
                        "values": []
                    }
                }
            }"""


class ESAdapter(KeyValueStorage):
    def __init__(self, es_endpoint='http://localhost:9200'):
        KeyValueStorage.__init__(self)

        self.es_endpoint = es_endpoint

    def get_record(self, record_id, table_name):
        # table_name = index in this case
        if not isinstance(record_id, list):
            record_id = [record_id]
        query = json.loads(query_str)
        query['query']['ids']['values'] = record_id
        url = '{}/{}/_search'.format(self.es_endpoint, table_name)
        r = requests.post(url, data=json.dumps(query))
        sources = list()
        for hit in r.json()['hits']['hits']:
            sources.append(hit['_source'])
        return sources

    def insert_record(self, record_id, record, table_name):
        print('lol wat')

    def create_table(self, table_name):
        print('no way')

    def tables(self):
        print('thats not how it works')

    def insert_records_batch(self, records, table_name):
        print('see insert_record')
