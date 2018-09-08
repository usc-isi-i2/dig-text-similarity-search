import os
import sys
from time import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from digtextsimilaritysearch.indexer.IVF16K_index_handler \
    import DiskBuilderIVF16K
from digtextsimilaritysearch.vectorizer.sentence_vectorizer \
    import SentenceVectorizer
from digtextsimilaritysearch.storage.es_adapter \
    import ESAdapter
from digtextsimilaritysearch.process_documents.document_processor \
    import DocumentProcessor


t_start = time()

# Resource paths
cwd = os.getcwd()
emb_dir = os.path.join(cwd, 'data/vectorized_sage_news/new_2018-08-from07to13')
index_dir = os.path.join(cwd, 'saved_indexes/IVF16K_indexes')
subindex_dir = os.path.join(index_dir, 'subindexes')

# Get .npz paths
small_npzs = list()
for (dir_path, _, file_list) in os.walk(emb_dir):
    for f in file_list:
        if f.startswith('vect') and f.endswith('.npz'):
            small_npzs.append(os.path.join(dir_path, f))
    break
small_npzs.sort()

# Make invlist paths
small_invlists = list()
for npz in small_npzs:
    invlist_name = 'invl_' + str(npz.split('/')[-1]).replace('.npz', '.index')
    small_invlists.append(os.path.join(subindex_dir, invlist_name))


# Init
t_init0 = time()
empty_index_path = os.path.join(index_dir, 'emptyTrainedIVF16384.index')
idx_bdr = DiskBuilderIVF16K(path_to_empty_index=empty_index_path)

sv = SentenceVectorizer()

logstash_path = '/lfs1/dig_text_sim/IVF16K_logstash_input.jl'
es = ESAdapter(logstash_file_path=logstash_path)

table = 'dig-text-similarity-search-IVF16K'
dp = DocumentProcessor(indexer=None, index_builder=idx_bdr,
                       vectorizer=sv, storage_adapter=es,
                       table_name=table)
t_init1 = time()
print('\n\nInitialized in {:0.2f}s\n'.format(t_init1-t_init0))

print('{} .npz files found\n'.format(len(small_npzs)))

# DoIT!
t_0 = time()
timestamps = list()
timestamps.append(0)
doit = False if len(sys.argv) > 1 else True
if doit:
    for i, (npz, invl) in enumerate(zip(small_npzs, small_invlists)):
        t_1 = time()
        try:
            dp.index_docs_on_disk(offset=(i*100000),
                                  path_to_npz=npz,
                                  path_to_invlist=invl)
        except Exception as e:
            print(e)
        timestamps.append(time()-t_1)
        if i % 20 == 0 or i >= len(small_npzs)-2:
            print('  {:4d} of {} .npz files indexed'.format(i+1, len(small_npzs)))
            print('  Average time per chunk: {:0.2f}s'
                  '\n'.format(sum(timestamps[1:])/len(timestamps[1:])))
else:
    dp.index_builder.extend_invlist_paths(small_invlists)

# Merge
merged_ivfs = os.path.join(index_dir, 'mergedIVF16384.ivfdata')
deployable_index = os.path.join(index_dir, 'populatedIVF16384.index')
ntotal = dp.build_index_on_disk(merged_ivfs_path=merged_ivfs,
                                merged_index_path=deployable_index)
t_end = time()
print('* Indexes merged in {:0.2f}s'.format(t_end-t_0-sum(timestamps)))
m, s = divmod(t_end-t_start, 60)
print('\n\nIndexing completed in {}m{:0.2f}s'.format(int(m), s))
print('Number of chunks indexed: {}'.format(len(small_npzs)))
print('Number of vectors indexed: {}'.format(ntotal))
