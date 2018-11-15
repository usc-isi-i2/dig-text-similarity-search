import os

saved_indexes = os.path.abspath('../saved_indexes/')

config = {
    "faiss_index_path": saved_indexes,
    "hbase_server": "localhost",
    "host": "0.0.0.0",
    "port": "5954",
    "es_endpoint": "http://mydig-sage-internal.isi.edu/es",
    "es_index": "sage_news_v2"

}
