import os

from time import time

from digtextsimilaritysearch.indexer.faiss_indexer \
    import FaissIndexer
from digtextsimilaritysearch.vectorizer.sentence_vectorizer \
    import SentenceVectorizer
from digtextsimilaritysearch.storage.es_adapter \
    import ESAdapter
from digtextsimilaritysearch.process_documents.document_processor \
    import DocumentProcessor


# Init
t_start = time()
sv = SentenceVectorizer()

cwd = os.getcwd()
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


# Test Queries
q0 = 'Before 8 September 2018, will the UK request an extension ' \
     'to Article 50 for leaving the EU?'

q1 = 'Will China execute or be targeted in an acknowledged ' \
     'national military attack before 1 September 2018?'

q2 = 'Will the WHO declare a Public Health Emergency of ' \
     'International Concern (PHEIC) before 1 September 2018?'

q3 = 'Will UK Prime Minister Theresa May announce her resignation, ' \
     'lose a confidence vote, or otherwise vacate her office ' \
     'before 8 September 2018?'

q4 = 'Will American pastor Andrew Brunson leave Turkey ' \
     'before 8 September 2018?'

queries = [q0, q1, q2, q3, q4]


# Run it
all_results = list()
time_stamps = list()
for q in queries:
    t_0 = time()
    results = dp.query_text(str_query=q, k=10)
    all_results.append(results)
    t_diff = time() - t_0
    time_stamps.append(t_diff)


# Report it
for i, (q, rs, t) in enumerate(zip(queries, all_results, time_stamps), start=1):
    file_name = 'query_test_' + str(i) + '_results.txt'
    file_path = os.path.join(cwd, file_name)

    try:
        with open(file_path, 'x') as f:
            f.write('Query: {}\n'.format(q))
            f.write('Results were gathered in: {:0.4f}s\n\n'.format(t))
            for j, r in enumerate(rs, start=1):
                f.write(' Result: {}\n'.format(j))
                f.write(' Difference Score: {:0.5f}\n'.format(r['score']))
                f.write(' Text: {}\n'.format(r['sentence']))
                f.write(' Document ID: {}\n\n'.format(r['doc_id']))

    except FileExistsError:
        print('File already exists: {}'.format(file_path))
