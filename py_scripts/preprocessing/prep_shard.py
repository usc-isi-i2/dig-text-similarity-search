# <editor-fold desc="Basic Imports">
import os
import os.path as p
import requests
from time import time
from argparse import ArgumentParser

import sys
sys.path.append(p.join(p.dirname(__file__), '..'))
sys.path.append(p.join(p.dirname(__file__), '../..'))
# </editor-fold>

# <editor-fold desc="Parse Command Line Args">
prog_file_path = p.join(p.dirname(__file__), 'progress.txt')
relative_base_path = '../../base_indexes/USE_lite_base_IVF16K.index'
base_index_path = p.abspath(p.join(p.dirname(__file__), relative_base_path))

arp = ArgumentParser(description='Vectorize Sentences for Searchable Index.')

arp.add_argument('input_dir', help='Path to raw news dir.')
arp.add_argument('output_dir', help='Path to saved index dir.')
arp.add_argument('-p', '--progress_file', default=prog_file_path,
                 help='For keeping track of news that has been preprocessed. '
                      'Default: dig-text-similarity-search/progress.txt')
arp.add_argument('-b', '--base_index_path', default=base_index_path,
                 help='Path to pre-trained empty faiss index. '
                      'Default: dig-text-similarity-search/base_indexes/*.index')
arp.add_argument('-l', '--large', action='store_true',
                 help='Toggle large Universal Sentence Encoder (Transformer NN).')
arp.add_argument('-m', '--m_per_batch', type=int, default=512*128,
                 help='Sentences per batch.')
arp.add_argument('-n', '--n_per_minibatch', type=int, default=64,
                 help='Sentences per mini-batch.')
arp.add_argument('-v', '--verbose', action='store_true',
                 help='Shows progress of batch vectorization.')
arp.add_argument('-t', '--num_threads', default='2',
                 help='Set CPU thread budget for numpy.')
arp.add_argument('-d', '--no_delete', action='store_false', default=True,
                 help='Keeps faiss indexes for each batch after merging on-disk.')
arp.add_argument('-a', '--add_shard', action='store_true',
                 help='Adds shard to running similarity server.')
arp.add_argument('-u', '--url', default='http://localhost:5954/faiss',
                 help='Port handling similarity server.')
arp.add_argument('-T', '--TF_logging', action='store_false', default=True,
                 help='Increase verbosity of TensorFlow.')
opts = arp.parse_args()
# </editor-fold>

if opts.num_threads:
    print('\nRestricting numpy to {} thread(s)\n'.format(opts.num_threads))
    os.environ['OPENBLAS_NUM_THREADS'] = opts.num_threads
    os.environ['NUMEXPR_NUM_THREADS'] = opts.num_threads
    os.environ['MKL_NUM_THREADS'] = opts.num_threads
    os.environ['OMP_NUM_THREADS'] = opts.num_threads

from dt_sim.data_reader.jl_io_funcs import check_all_docs, get_all_docs
from dt_sim.data_reader.misc_io_funcs import check_unique, clear_dir
from dt_sim.vectorizer.sentence_vectorizer import SentenceVectorizer
from dt_sim.indexer.index_builder import OnDiskIVFBuilder
from dt_sim.processor.corpus_processor import CorpusProcessor

# Suppress TF logging
if opts.TF_logging:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Init
sv = SentenceVectorizer(large=opts.large)
idx_bdr = OnDiskIVFBuilder(path_to_base_index=opts.base_index_path)
cp = CorpusProcessor(vectorizer=sv, index_builder=idx_bdr,
                     progress_file=opts.progress_file)

# Track progress
prepped_news = cp.track_preprocessing(cp.progress_file, verbose=opts.verbose)
raw_news = cp.get_news_paths(opts.input_dir, verbose=opts.verbose)
candidates = cp.candidate_files(prepped_news, raw_news, verbose=opts.verbose)
file_to_process = candidates[:1]   # Preprocesses one news.jl per call


def main(raw_jl, output_dir: str = opts.output_dir,
         m_per_batch: int = opts.m_per_batch, n_per_minibatch: int = opts.n_per_minibatch,
         no_delete: bool = opts.no_delete, verbose: bool = opts.verbose,
         add_shard: bool = opts.add_shard, url: str = opts.url):

    subidx_dir, shard_date = cp.init_paths(raw_jl)
    if verbose:
        print('Will process: {}\n'.format(raw_jl))

    # Check File Content
    if verbose:
        print('\nReading file: {}'.format(raw_jl))

    jl_stats = check_all_docs(raw_jl, batch_size=m_per_batch)
    (doc_count, line_count, junk, n_batches) = jl_stats
    if verbose:
        print('* Found {} good documents with {} total sentences\n'
              '* Will skip {} junk documents\n'
              '* Processing {} batches\n'
              ''.format(doc_count, line_count, junk, n_batches))

    # Preprocess
    t_start = time()
    doc_batch_gen = get_all_docs(raw_jl, batch_size=m_per_batch)
    for i, (batched_sents, batched_ids) in enumerate(doc_batch_gen):
        t_0 = time()
        if verbose:
            print('  Starting doc batch:  {:3d}'.format(i+1))

        subidx = str(raw_jl.split('/')[-1]).replace('.jl', '_{:03d}_sub.index'.format(i))
        subidx_path = p.join(subidx_dir, subidx)

        if p.exists(subidx_path):
            print('  File exists: {} \n'
                  '  Skipping...  '.format(subidx_path))
            cp.index_builder.include_subidx_path(subidx_path)
        else:
            # Vectorize
            emb_batch, id_batch = cp.batch_vectorize(
                text_batch=batched_sents, id_batch=batched_ids,
                n_minibatch=n_per_minibatch, very_verbose=False
            )
            t_vect = time()
            if verbose:
                print('  * Vectorized in {:6.2f}s'.format(t_vect - t_0))

            # Make faiss subindex
            subidx_path = check_unique(subidx_path)
            cp.index_builder.generate_subindex(subidx_path, emb_batch, id_batch)
            t_subidx = time()
            if verbose:
                print('  * Subindexed in {:6.2f}s'.format(t_subidx - t_vect))

            # Clear graph
            del emb_batch, batched_sents, id_batch
            cp.vectorizer.close_session()
            t_reset = time()
            if verbose:
                print('  * Cleared TF in {:6.2f}s'.format(t_reset - t_subidx))

            # Restart TF session if necessary
            if i < n_batches - 1:
                cp.vectorizer.start_session()
                if verbose:
                    print('  * Started TF in {:6.2f}s'.format(time() - t_reset))

        if verbose:
            mp, sp = divmod(time() - t_start, 60)
            print('  Completed doc batch: {:3d}/{}      '
                  '  Total time passed: {:3d}m{:0.2f}s\n'
                  ''.format(i+1, n_batches, int(mp), sp))

    # Merge
    # TODO: Title indexes
    t_merge = time()
    merged_index_path = shard_date + '_all.index'
    merged_index_path = p.join(output_dir, merged_index_path)
    merged_index_path = check_unique(merged_index_path)
    merged_ivfdata_path = shard_date + '_all.ivfdata'
    merged_ivfdata_path = p.join(output_dir, merged_ivfdata_path)
    merged_ivfdata_path = check_unique(merged_ivfdata_path)
    if verbose:
        print('\n  Merging {} on-disk'.format(merged_index_path.split('/')[-1]))

    assert cp.index_builder.index_path_clear(merged_index_path)
    assert cp.index_builder.index_path_clear(merged_ivfdata_path, '.ivfdata')

    n_vect = cp.index_builder.merge_IVFs(index_path=merged_index_path,
                                         ivfdata_path=merged_ivfdata_path)

    if verbose:
        mm, sm = divmod(time() - t_merge, 60)
        print('  Merged subindexes ({} vectors) in: {:3d}m{:0.2f}s'
              ''.format(n_vect, int(mm), sm))

    # Record progress
    cp.record_progress(raw_jl)

    # Clear sub.index files after merge
    if no_delete:
        clear_dir(subidx_dir)
        if verbose:
            print('\n  Cleared sub.index files')

    if add_shard:
        try:
            url = url
            payload = {'path': merged_index_path}
            r = requests.put(url, params=payload)
            print(r.text)
        except Exception as e:
            print('Shard was not added because an exception occurred: {}'.format(e))


if __name__ == '__main__':
    if len(file_to_process):
        jl = file_to_process[0]
        main(raw_jl=jl)
    else:
        print('Nothing to process.')
