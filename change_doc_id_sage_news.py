from kafka import KafkaProducer
from kafka import KafkaConsumer
import json
import signal
import datetime
import requests

broker_list = [
    'kafka:9092'
]

args = {
}

# latest_doc_id_file_path = 'latest_doc_id_value.txt'
# latest_doc_id = int(open(latest_doc_id_file_path).readlines()[0])
es_url = 'http://54.193.27.186:9200'
es_index = 'sage_news_v2'
query = {"_source": "doc_id",

         "query": {
             "match_all": {}
         },
         "sort": [
             {
                 "timestamp_crawl": {
                     "order": "desc"
                 }
             }
         ],
         "size": 1

         }


def get_latest_doc_id():
    response = requests.post(url='{}/{}/_search'.format(es_url, es_index), json=query)
    return response.json()['hits']['hits'][0]['_source']['doc_id']


news_output_path = '/data/sage_news_backup'

todays_date = str(datetime.date.today())
news_output_file = open('{}/{}.jl'.format(news_output_path, todays_date), mode='a', encoding='utf-8')

producer = KafkaProducer(
    bootstrap_servers=broker_list,
    value_serializer=lambda v: v.encode('utf-8'),
    **args
)

timeout = 3600

consumer = KafkaConsumer(
    'sage_news_v2_out',
    bootstrap_servers=broker_list,
    group_id='doc_id_manipulation',
    auto_offset_reset='earliest',
    **args
)


def flush_out_stuff():
    # global latest_doc_id_file_path
    global news_output_path
    # global latest_doc_id
    global news_output_file
    # open(latest_doc_id_file_path, mode='w').write(str(latest_doc_id))

    try:
        news_output_file.close()
    except:
        pass

    news_output_file = open('{}/{}.jl'.format(news_output_path, str(datetime.date.today())), mode='a', encoding='utf-8')


# Register an handler for the timeout
def handler(signum, frame):
    flush_out_stuff()

    signal.alarm(timeout)


# Register the signal function handler
signal.signal(signal.SIGALRM, handler)

signal.alarm(timeout)


def read_message(consumer):
    latest_doc_id = int(get_latest_doc_id())
    global news_output_file
    for msg in consumer:
        try:
            doc = json.loads(msg.value.decode('utf-8'))
            # infinite loop alert
            if doc.get('type', '') != 'sage_news_v2':
                doc['doc_id'] = str(latest_doc_id + 1)
                latest_doc_id += 1
                doc['type'] = 'sage_news_v2'
                doc_str = json.dumps(doc)

                r = producer.send('sage_news_v2_out', doc_str)
                r.get(timeout=60)
                news_output_file.write(doc_str + '\n')
        except:
            flush_out_stuff()
            pass


read_message(consumer)
