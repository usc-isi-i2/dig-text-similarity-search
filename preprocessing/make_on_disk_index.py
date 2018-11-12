# <editor-fold desc="Imports">
import os
import sys
from time import time
from optparse import OptionParser
# <editor-fold desc="Parse Options">
cwd = os.path.abspath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
base_index_path = os.path.abspath(os.path.join(cwd, '../saved_indexes/IVF16K_indexes/'
                                                    'emptyTrainedIVF16384.index'))

arg_parser = OptionParser()
arg_parser.add_option('-i', '--input_npz_dir')
arg_parser.add_option('-o', '--output_index_dir')
arg_parser.add_option('-s', '--subindex_dir')
arg_parser.add_option('-b', '--base_empty_index', default=base_index_path)
arg_parser.add_option('-m', '--merged_ivf_data', default='mergedIVF16384.ivfdata')
arg_parser.add_option('-p', '--populated_index', default='populatedIVF16384.index')
arg_parser.add_option('-e', '--build_from_existing', action='store_true', default=False)
arg_parser.add_option('-n', '--n_subindexes', type='int', default=-1)
(args, _) = arg_parser.parse_args()
# </editor-fold>

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from dt_sim_api.indexer.IVF_disk_index_handler import DiskBuilderIVF
from dt_sim_api.vectorizer import SentenceVectorizer
from dt_sim_api.process_documents.document_processor import DocumentProcessor
# </editor-fold>


"""
Script for making many, small subindexes and merges them into an 
on-disk searchable index. 

  - Requires sentences to be preprocessed into .npz files. 
  - Makes one subindex per .npz
  - Makes faiss_ids with dummy offset (for demonstration purposes)

Required Args: 
    -i  Path to .npz directory
    -o  Path to index directory
    -s  Path to subindex directory 
    
Options:
    -b  Path to pre-trained, base index (empty)
    -m  Name of on-disk, searchable IVF data (this is what is searched)
    -p  Name of populated index file (this is linked to IVF data file)
    -e  Builds on-disk index from existing subindexes (default False)
    -n  Number of subindexes to merge, sorted from most recent (default all)
"""


t_start = time()

# Resource paths
emb_dir = args.input_npz_dir
assert os.path.isdir(emb_dir), 'Full path does not exist: {}'.format(emb_dir)
index_dir = args.output_index_dir
assert os.path.isdir(index_dir), 'Full path does not exist: {}'.format(index_dir)
subindex_dir = args.subindex_dir
assert os.path.isdir(subindex_dir), 'Full path does not exist: {}'.format(subindex_dir)

# Get .npz paths
small_npzs = list()
for (dir_path, _, file_list) in os.walk(emb_dir):
    for f in file_list:
        if f.startswith('vect') and f.endswith('.npz'):
            small_npzs.append(os.path.join(dir_path, f))
    break
small_npzs.sort(reverse=True)

# Make invlist paths
small_invlists = list()
for npz in small_npzs:
    invlist_name = 'invl_' + str(npz.split('/')[-1]).replace('.npz', '.index')
    small_invlists.append(os.path.join(subindex_dir, invlist_name))


# Init
t_init0 = time()
empty_index_path = args.base_empty_index
assert os.path.exists(empty_index_path), 'Faiss index builder requires an empty pretrained index'
idx_bdr = DiskBuilderIVF(path_to_empty_index=empty_index_path)

sv = SentenceVectorizer

table = 'dig-text-similarity-search-IVF16K'
dp = DocumentProcessor(indexer=None, index_builder=idx_bdr,
                       vectorizer=sv, storage_adapter=None)
t_init1 = time()
print('\n\nInitialized in {:0.2f}s\n'.format(t_init1-t_init0))

print('{} .npz files found\n'.format(len(small_npzs)))

# DoIT!
t_0 = time()
timestamps = list()
timestamps.append(0)
if args.build_from_existing:
    # Add paths to subindexes to be merged
    dp.index_builder.extend_invlist_paths(small_invlists)

    # Merge
    merged_ivfs = os.path.join(index_dir, args.merged_ivf_data)
    deployable_index = os.path.join(index_dir, args.populated_index)
    ntotal = dp.build_index_on_disk(merged_ivfs_path=merged_ivfs,
                                    merged_index_path=deployable_index)

    t_end = time()
    print('\nNumber of vectors indexed: {}'.format(ntotal))
    mm, sm = divmod(t_end - t_0, 60)
    print('* Subindexes merged in {}m{:0.2f}s'.format(int(mm), sm))

else:
    for i, (npz, invl) in enumerate(zip(small_npzs, small_invlists)):
        if not os.path.exists(invl):
            t_1 = time()
            dp.index_docs_on_disk(path_to_npz=npz, path_to_invlist=invl)
            timestamps.append(time()-t_1)
        else:
            print('  Skipping: {}'.format(invl))

        if i % 50 == 0 or i >= len(small_npzs)-2:
            print('  {:4d} of {} .npz files indexed'.format(i, len(small_npzs)))
            print('  Average time per chunk: {:0.2f}s'
                  '\n'.format(sum(timestamps[1:])/len(timestamps[1:])))

    t_end = time()
    print('\nNumber of subindexes created: {}'.format(len(small_npzs)))
    ms, ss = divmod(t_end - t_0, 60)
    print('* Subindexes made in {}m{:0.2f}s'.format(int(ms), ss))

m_tot, s_tot = divmod(t_end-t_start, 60)
print('\n* Indexing completed in {}m{:0.2f}s'.format(int(m_tot), s_tot))
