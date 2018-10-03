from kafka import KafkaProducer
from kafka import KafkaConsumer
import json
import signal
import datetime

broker_list = [
    'kafka:9092'
]

args = {
}

latest_doc_id_file_path = 'latest_doc_id_value.txt'
latest_doc_id = int(open(latest_doc_id_file_path).readlines()[0])
news_output_path = 'some_other_path'

todays_date = str(datetime.date.today())
news_output_file = open('{}/{}'.format(news_output_path, todays_date), mode='a', encoding='utf-8')

producer = KafkaProducer(
    bootstrap_servers=broker_list,
    value_serializer=lambda v: v.encode('utf-8'),
    **args
)

timeout = 3600

consumer = KafkaConsumer(
    'sage_news_v2_in',
    bootstrap_servers=broker_list,
    group_id='doc_id_manipulation',
    auto_offset_reset='earliest',
    **args
)


# Register an handler for the timeout
def handler(signum, frame):
    global latest_doc_id_file_path
    global news_output_path
    global latest_doc_id
    global news_output_file
    open(latest_doc_id_file_path, mode='w').write(str(latest_doc_id))

    try:
        news_output_file.close()
    except:
        pass

    news_output_file = open('{}/{}'.format(news_output_path, str(datetime.date.today())), mode='a', encoding='utf-8')

    signal.alarm(timeout)


# Register the signal function handler
signal.signal(signal.SIGALRM, handler)

signal.alarm(3600)


def read_message(consumer):
    global latest_doc_id
    global news_output_file
    for msg in consumer:
        doc = json.loads(msg.value.decode('utf-8'))
        doc['doc_id'] = str(latest_doc_id + 1)
        latest_doc_id += 1
        doc['type'] = 'sage_news_v2'
        doc_str = json.dumps(doc)

        r = producer.send('sage_news_v2_out', doc_str)
        r.get(timeout=60)
        news_output_file.write(doc_str + '\n')


read_message(consumer)
