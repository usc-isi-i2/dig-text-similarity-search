from kafka import KafkaProducer
from kafka import KafkaConsumer
import json
import datetime

broker_list = [
    'kafka:9092'
]

args = {
}

news_output_path = '/data/sage_news_backup'

todays_date = str(datetime.date.today())
news_output_file = open('{}/{}.jl'.format(news_output_path, todays_date), mode='a', encoding='utf-8')

producer = KafkaProducer(
    bootstrap_servers=broker_list,
    value_serializer=lambda v: v.encode('utf-8'),
    **args
)

consumer = KafkaConsumer(
    'sage_news_v2_out',
    bootstrap_servers=broker_list,
    group_id='doc_id_manipulation',
    auto_offset_reset='earliest',
    **args
)


def read_message(consumer):
    for msg in consumer:
        try:
            doc = json.loads(msg.value.decode('utf-8'))
            # infinite loop alert
            if doc.get('type', '') != 'sage_news_v2':
                doc['type'] = 'sage_news_v2'
                doc_str = json.dumps(doc)

                r = producer.send('sage_news_v2_out', doc_str)
                r.get(timeout=60)
                news_output_file.write(doc_str + '\n')

        except:

            pass


read_message(consumer)
