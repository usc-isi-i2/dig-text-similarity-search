import os

from digtextsimilaritysearch.indexer.faiss_indexer \
    import FaissIndexer
from digtextsimilaritysearch.vectorizer.sentence_vectorizer \
    import SentenceVectorizer
from digtextsimilaritysearch.storage.hbase_adapter \
    import HBaseAdapter
from digtextsimilaritysearch.process_documents.document_processor \
    import DocumentProcessor


cwd = os.getcwd()
emb_dir = os.path.join(cwd, 'data/vectorized_sage_news')
news_npzs = ['new_2018-08-07',
             'new_2018-08-08',
             'new_2018-08-09',
             'new_2018-08-10',
             'new_2018-08-11',
             'new_2018-08-12',
             'new_2018-08-13']

sv = SentenceVectorizer()
hb = HBaseAdapter('localhost')
fi = FaissIndexer()

idx_name = 'FlatL2_Aug_test.index'
idx_path = os.path.join(cwd, 'saved_indexes', idx_name)

dp = DocumentProcessor(indexer=fi, vectorizer=sv, hbase_adapter=hb,
                       index_save_path=idx_path)

for j, npz in enumerate(news_npzs):

    if j == 1:

        npz_file = os.path.join(emb_dir, npz + '.npz')
        print('Loading {}'.format(npz_file))

        dp.vector_save_path = npz_file

        print('Adding to index...')
        dp.index_documents(load_vectors=True)
