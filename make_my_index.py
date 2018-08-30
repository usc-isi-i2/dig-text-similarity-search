import os
import sys

from time import time, sleep

from digtextsimilaritysearch.indexer.faiss_indexer \
    import FaissIndexer
from digtextsimilaritysearch.vectorizer.sentence_vectorizer \
    import SentenceVectorizer
from digtextsimilaritysearch.storage.es_adapter \
    import ESAdapter
from digtextsimilaritysearch.process_documents.document_processor \
    import DocumentProcessor


t_start = time()

cwd = os.getcwd()
emb_dir = os.path.join(cwd, 'data/vectorized_sage_news/new_2018-08-from07to13')
completed_log = os.path.join(emb_dir, 'completed.txt')


def read_completed():
    completed = list()
    with open(completed_log, 'r') as clog:
        for line in clog:
            completed.append(str(line).split('\n')[0])
    return set(completed)


def mark_completed(completed_npz_path):
    with open(completed_log, 'a') as clog:
        clog.write(completed_npz_path + '\n')


# get paths
small_npzs = list()
for (dir_path, _, file_list) in os.walk(emb_dir):
    for f in file_list:
        if f.endswith('.npz'):
            small_npzs.append(os.path.join(dir_path, f))
    break
small_npzs.sort()


# refine queue
if os.path.isfile(completed_log):
    npz_queue = list(set(small_npzs) - read_completed())
    npz_queue.sort()
    print('y')
else:
    npz_queue = small_npzs
    print('n')

print(small_npzs == npz_queue)
if small_npzs == npz_queue and os.path.isfile(completed_log):
    print(read_completed())


# Init
sv = SentenceVectorizer()

idx_name = 'FlatL2_Aug_7-13_ES.index'
idx_path = os.path.join(cwd, 'saved_indexes', idx_name)
fi = FaissIndexer(path_to_index_file=idx_path)

logstash_path = '/lfs1/dig_text_sim/logstash_input.jl'
es = ESAdapter(logstash_file_path=logstash_path)

table = 'dig-text-similarity-search'
dp = DocumentProcessor(indexer=fi, vectorizer=sv, storage_adapter=es,
                       index_save_path=idx_path, table_name=table)

t_init = time()
print('\nTime used for initialization: {}s'.format(t_init-t_start))


# Run it
print('\n\n{}/{} .npz file chunks to add to index'
      ''.format(len(npz_queue), len(small_npzs)))
time_stamps = list()
for i, npz in enumerate(npz_queue, start=(len(small_npzs)-len(npz_queue))):
    t_0 = time()

    print('\nLoading {}'.format(npz))
    dp.vector_save_path = npz

    dp.index_documents(load_vectors=True,
                       save_faiss_index=True)
    print('{} added to index'.format(npz))

    t_diff = time()-t_0
    time_stamps.append(t_diff)
    print('Chunk {} complete'.format(i))
    print('Time passed: {}s'.format(t_diff))
    print('Avg time per chunk: {}'.format(sum(time_stamps)/len(time_stamps)))

    mark_completed(npz)

    sleep(1.5)
