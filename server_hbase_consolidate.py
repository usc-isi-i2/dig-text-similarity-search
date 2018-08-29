import os
import sys

from time import time

from digtextsimilaritysearch.indexer.faiss_indexer \
    import FaissIndexer
from digtextsimilaritysearch.vectorizer.sentence_vectorizer \
    import SentenceVectorizer
from digtextsimilaritysearch.storage.hbase_adapter \
    import HBaseAdapter
from digtextsimilaritysearch.process_documents.document_processor \
    import DocumentProcessor


# Note: ensure hbase docker is running
# docker pull dajobe/hbase
# docker run -d -p 9090:9090 -p 2181:2181 -v /lfs1/dig/hbase_storage:/data dajobe/hbase

t_start = time()

cwd = os.getcwd()
emb_dir = os.path.join(cwd, 'data/vectorized_sage_news')
news_dirs = ['new_2018-08-07',
             'new_2018-08-08',
             'new_2018-08-09',
             'new_2018-08-10',
             'new_2018-08-11',
             'new_2018-08-12',
             'new_2018-08-13']

news_npzs = list()
for d in news_dirs:
    files = list()
    for (dir_path, _, file_names) in os.walk(os.path.join(emb_dir, d)):
        files.extend(file_names)
        for f in files:
            news_npzs.append(os.path.join(dir_path, f))
        continue
news_npzs.sort()

sv = SentenceVectorizer()

idx_name = 'FlatL2_Aug_7-13_' + sys.argv[1] + '.index'
idx_path = os.path.join(cwd, 'saved_indexes', idx_name)
fi = FaissIndexer(path_to_index_file=idx_path)

hb = HBaseAdapter('localhost')

dp = DocumentProcessor(indexer=fi, vectorizer=sv, storage_adapter=hb,
                       index_save_path=idx_path)

t_init = time()
print('\nTime used for initialization: {}s'.format(t_init-t_start))
time_stamps = list()

print('\n\n{} .npz file chunks to add to index'.format(len(news_npzs)))
s = int(sys.argv[2]) if len(sys.argv) >= 3 else 0
e = int(sys.argv[3]) if len(sys.argv) >= 4 else -1
for npz in news_npzs[s:e]:
    t_0 = time()

    print('\nLoading {}'.format(npz))
    dp.vector_save_path = npz

    try:
        dp.index_documents(load_vectors=True,
                           save_faiss_index=True,
                           batch_mode=True,
                           batch_size=1000)
        print('{} added to index'.format(npz))

    except Exception as e:
        print(e)
        pass

    finally:
        t_1 = time()
        t_diff = t_1-t_0
        time_stamps.append(t_diff)
        print('Time passed: {}s'.format(t_diff))
