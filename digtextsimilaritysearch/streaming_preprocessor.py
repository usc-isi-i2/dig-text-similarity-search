import os
import json
import numpy as np

from kafka import KafkaConsumer
from .indexer.IVF_disk_index_handler import DiskBuilderIVF
from .vectorizer.sentence_vectorizer import SentenceVectorizer
from .process_documents.document_processor import DocumentProcessor


# Funcs
def aggregate_docs(doc_stream, b_size=5000):
    batched_ids = list()
    batched_text = list()
    for ii, document in enumerate(doc_stream, start=1):
        document = json.loads(document.value)
        content = document['lexisnexis']['doc_description']
        if content and not content == '' and not content == 'DELETED_STORY':
            text = list()
            text.append(document['lexisnexis']['doc_title'])
            text.extend(document['split_sentences'])

            sent_ids = list()
            doc_id = document['doc_id']
            base_sent_id = np.int64(doc_id + '0000')
            for jj, _ in enumerate(text):
                sent_ids.append(base_sent_id + jj)
            sent_ids = np.vstack(sent_ids).astype(np.int64)
            assert sent_ids.shape[0] == len(text), \
                'Something went wrong while making sent_ids'

            batched_text.extend(text)
            batched_ids.append(sent_ids)

            # TODO: Add condition to flush docs after timer
            if ii % b_size == 0:
                batched_ids = np.vstack(batched_ids).astype(np.int64)
                yield batched_text, batched_ids
                batched_text = list()
                batched_ids = list()


# Init paths
# TODO: Set paths with command line args
npz_dir = 'EDIT_ME'
subidx_dir = 'EDIT_ME'
empty_index_path = 'EDIT_ME'


# Init DocumentProcessor
idx_bdr = DiskBuilderIVF(path_to_empty_index=empty_index_path)

sv = SentenceVectorizer()

table = 'EDIT_ME'
dp = DocumentProcessor(indexer=None, index_builder=idx_bdr,
                       vectorizer=sv, storage_adapter=None,
                       table_name=table)


# Init Kafka
broker_list = ['EDIT_ME']

args = {}

consumer = KafkaConsumer('EDIT_ME',
                         bootstrap_servers=broker_list,
                         group_id='EDIT_ME',
                         auto_offset_reset='earliest',
                         **args)


# Passive Consumer
def main():
    # TODO: Set batch_size with command line args
    batch_size = 5000
    doc_batch_gen = aggregate_docs(consumer, b_size=batch_size)
    for batched_sents, batched_ids in doc_batch_gen:

        # Vectorize
        batched_embs = dp.vectorizer.make_vectors(batched_sents)

        # Save to .npz
        # TODO: Make unique .npz paths
        npz = 'EDIT_ME'
        npz_path = os.path.join(npz_dir, npz)
        dp.vectorizer.save_with_ids(file_path=npz_path,
                                    embeddings=batched_embs,
                                    sentences=batched_sents,
                                    sent_ids=batched_ids)

        # Make faiss subindex
        # TODO: Make unique subindex path
        subidx = 'EDIT_ME'
        subidx_path = os.path.join(subidx_dir, subidx)
        dp.index_docs_on_disk(path_to_npz=npz_path,
                              path_to_invlist=subidx_path)


if __name__ == '__main__':
    main()
