import os
import json
import numpy as np
from indexer.IVF_disk_index_handler import DiskBuilderIVF
from vectorizer.sentence_vectorizer import SentenceVectorizer
from process_documents.document_processor import DocumentProcessor
from time import time
from optparse import OptionParser
# <editor-fold desc="Parse Command Line Options">
options = OptionParser()
options.add_option('-i', '--input_filepath')
options.add_option('-n', '--npz_dir')
options.add_option('-s', '--subindex_dir')
options.add_option('-b', '--base_index_path')
options.add_option('-m', '--m_per_batch', type='int', default=250000)
(opts, _) = options.parse_args()
# </editor-fold>


"""
Options:
    -i  Full path to input .jl file
    -n  Full path to dir for writing .npz files
    -s  Full path to dir for writing .index files 
    -b  Full path to empty, pre-trained faiss index
    -m  Minimum number of sentences per .npz/.index (default 250000)
"""


# Funcs
def aggregate_docs(file_path, b_size=250000):
    batched_text = list()
    batched_ids = list()
    with open(file_path, 'r') as f:
        for line in f:
            document = json.loads(line)

            content = document['lexisnexis']['doc_description']
            if content and not content == '' and not content == 'DELETED_STORY':
                text = list()
                text.append(document['lexisnexis']['doc_title'])
                text.extend(document['split_sentences'])

                doc_id = document['doc_id']
                base_sent_id = np.int64(doc_id + '0000')
                sent_ids = list()
                for jj, _ in enumerate(text):
                    sent_ids.append(base_sent_id + jj)
                sent_ids = np.vstack(sent_ids).astype(np.int64)
                assert sent_ids.shape[0] == len(text), \
                    'Something went wrong while making sent_ids'

                batched_text.extend(text)
                batched_ids.append(sent_ids)

            if len(batched_text) >= b_size:
                batched_ids = np.vstack(batched_ids).astype(np.int64)
                yield batched_text, batched_ids
                batched_text = list()
                batched_ids = list()

    batched_ids = np.vstack(batched_ids).astype(np.int64)
    yield batched_text, batched_ids


# Init paths
input_file = opts.input_filepath
batch_size = opts.m_per_batch
npz_dir = opts.npz_dir
subidx_dir = opts.subindex_dir
base_index_path = opts.base_index_path


# Init DocumentProcessor
idx_bdr = DiskBuilderIVF(path_to_empty_index=base_index_path)
sv = SentenceVectorizer()
dp = DocumentProcessor(indexer=None, index_builder=idx_bdr,
                       vectorizer=sv, storage_adapter=None)


# Preprocessing
def main():
    t_start = time()
    doc_batch_gen = aggregate_docs(file_path=input_file, b_size=batch_size)
    for i, (batched_sents, batched_ids) in enumerate(doc_batch_gen):

        # Vectorize
        batched_embs = dp.vectorizer.make_vectors(batched_sents)

        # Save to .npz
        npz = str(input_file.split('/')[-1]).replace('.jl', '_{:03d}.npz'.format(i))
        npz_path = os.path.join(npz_dir, npz)
        dp.vectorizer.save_with_ids(file_path=npz_path, embeddings=batched_embs,
                                    sentences=batched_sents, sent_ids=batched_ids)

        # Make faiss subindex
        subidx = 'subidx_' + str(npz.split('/')[-1]).replace('.npz', '.index')
        subidx_path = os.path.join(subidx_dir, subidx)
        dp.index_docs_on_disk(path_to_npz=npz_path, path_to_invlist=subidx_path)

        m, s = divmod(time() - t_start, 60)
        print('  Completed doc batch: {:3d}          '
              '  Time:{:3d}m{:0.2f}s'.format(i, int(m), s))


if __name__ == '__main__':
    main()
