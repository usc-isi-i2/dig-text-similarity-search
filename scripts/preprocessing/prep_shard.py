# <editor-fold desc="Basic Imports">
import os
import os.path as p
import re
import sys
import json
import datetime
import requests
from time import time
from optparse import OptionParser

sys.path.append(p.join(p.dirname(__file__), '..'))
sys.path.append(p.join(p.dirname(__file__), '../..'))
# </editor-fold>

# <editor-fold desc="Parse Command Line Options">
prog_file_path = p.join(p.dirname(__file__), 'progress.txt')
relative_base_path = '../../saved_indexes/USE_lite_base_IVF16K.index'
base_index_path = p.abspath(p.join(p.dirname(__file__), relative_base_path))

arg_parser = OptionParser()
arg_parser.add_option('-i', '--input_dir')
arg_parser.add_option('-o', '--output_dir')
arg_parser.add_option('-t', '--num_threads')
arg_parser.add_option('-p', '--progress_file', default=prog_file_path)
arg_parser.add_option('-b', '--base_index_path', default=base_index_path)
arg_parser.add_option('-l', '--large', action='store_true', default=False)
arg_parser.add_option('-m', '--m_per_batch', type='int', default=512*128)
arg_parser.add_option('-n', '--n_per_minibatch', type='int', default=32)
arg_parser.add_option('-r', '--report', action='store_true', default=False)
arg_parser.add_option('-d', '--delete_tmp_files', action='store_true', default=False)
arg_parser.add_option('-a', '--add_shard', action='store_true', default=False)
arg_parser.add_option('-u', '--url', default='http://localhost:5954/faiss')
arg_parser.add_option('-s', '--skip', type='int', default=0)
arg_parser.add_option('-T', '--TF_logging', action='store_false', default=True)
(opts, _) = arg_parser.parse_args()
# </editor-fold>

# Suppress TF logging
if opts.TF_logging:
    import tensorflow as tf
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if opts.num_threads:
    print('\nRestricting numpy to {} thread(s)\n'.format(opts.num_threads))
    os.environ['OPENBLAS_NUM_THREADS'] = opts.num_threads
    os.environ['NUMEXPR_NUM_THREADS'] = opts.num_threads
    os.environ['MKL_NUM_THREADS'] = opts.num_threads
    os.environ['OMP_NUM_THREADS'] = opts.num_threads

import numpy as np

from dt_sim_api.data_reader.jl_io_funcs import check_all_docs, get_all_docs
from dt_sim_api.data_reader.misc_io_funcs import check_unique, clear_dir
from dt_sim_api.vectorizer.sentence_vectorizer import SentenceVectorizer
from dt_sim_api.indexer.index_builder import LargeIndexBuilder
from dt_sim_api.processor.corpus_processor import CorpusProcessor


"""
To run this script:
    (from dig-text-similarity-search/)
    $ python scripts/preprocessing/prep_shard.py \ 
        -i {/path/to/split_sents_dir/} -o {/path/to/shard_index_dir/} -r -d

Options:
    -i  Path to raw news dir
    -o  Path to dir for writing merged, on-disk faiss index shard
    -t  Option to set thread budget for numpy to reduce CPU resource consumption 
            Useful if other tasks are running 
    -p  File to keep track of news that has already been processed 
            (default progress.txt)
    -b  Path to empty, pre-trained faiss index 
            (default ../saved_indexes/IVF16K_indexes/USE_lite_base_IVF16K.index)
    -m  Minimum number of sentences per batch 
            (default 512*128)
    -r  Bool to toggle prints 
            (default False)
    -d  Bool to delete intermediate .index files 
            (default False)
    -a  Bool to automatically add the created shard to the similarity server 
            (default False)
    -u  If -a is True, -u can be used to specify where to put() the new index 
            (default http://localhost:5954/faiss')
            * Note: url must end with '/faiss'

    -s  Development param: If preprocessing was interrupted after several 
            sub.index files were created, but before the on-disk shard was merged, 
            use -s <int:n_files_to_reuse> to reuse existing intermediate files. 
            * Note: Do NOT reuse partially created intermediate files
"""


# Init
sv = SentenceVectorizer(large=opts.large)
idx_bdr = LargeIndexBuilder(path_to_base_index=opts.base_index_path)
cp = CorpusProcessor(vectorizer=sv, index_builder=idx_bdr,
                     progress_file=opts.progress_file)

# Track progress
prepped_news = cp.track_preprocessing(cp.progress_file, verbose=opts.report)
raw_news = cp.get_news_paths(opts.input_dir, verbose=opts.report)
candidates = cp.candidate_files(prepped_news, raw_news, verbose=opts.report)
file_to_process = candidates[:1]   # Only preprocesses one news.jl


def main():
    raw_jl = file_to_process[0]
    subidx_dir, shard_date = cp.init_paths(raw_jl, opts.input_dir)
    if opts.report:
        print('Will process: {}\n'.format(raw_jl))

    # Check File Content
    if opts.report:
        print('\nReading file: {}'.format(raw_jl))

    jl_stats = check_all_docs(raw_jl, batch_size=opts.m_per_batch)
    (doc_count, line_count, junk, n_batches) = jl_stats
    if opts.report:
        print('* Found {} good documents with {} total sentences\n'
              '* Will skip {} junk documents\n'
              '* Processing {} batches\n'
              ''.format(doc_count, line_count, junk, n_batches))

    # Preprocess
    t_start = time()
    doc_batch_gen = get_all_docs(raw_jl, batch_size=opts.m_per_batch)
    for i, (batched_sents, batched_ids) in enumerate(doc_batch_gen):
        t_0 = time()
        if opts.report:
            print('  Starting doc batch:  {:3d}'.format(i+1))

        subidx = str(raw_jl.split('/')[-1]).replace('.jl', '_{:03d}_sub.index'.format(i))
        subidx_path = p.join(subidx_dir, subidx)

        if i < opts.skip:
            assert p.exists(subidx_path), \
                'Warning: File does not exist: {} \n' \
                'Aborting...'.format(subidx_path)
            cp.index_builder.include_subpath(subidx_path)

        elif p.exists(subidx_path):
            print('  File exists: {} \n'
                  '  Skipping...  '.format(subidx_path))
            cp.index_builder.include_subpath(subidx_path)

        else:
            # Vectorize
            emb_batch, id_batch = cp.batch_vectorize(
                text_batch=batched_sents, id_batch=batched_ids,
                n_minibatch=opts.n_per_minibatch, very_verbose=False
            )
            t_vect = time()
            if opts.report:
                print('  * Vectorized in {:6.2f}s'.format(t_vect - t_0))

            # Make faiss subindex
            subidx_path = check_unique(subidx_path)
            cp.index_builder.generate_subindex(subidx_path, emb_batch, id_batch)
            t_subidx = time()
            if opts.report:
                print('  * Subindexed in {:6.2f}s'.format(t_subidx - t_vect))

            # Clear graph
            del emb_batch, batched_sents, id_batch
            cp.vectorizer.close_session()
            t_reset = time()
            if opts.report:
                print('  * Cleared TF in {:6.2f}s'.format(t_reset - t_subidx))

            # Restart TF session if necessary
            if i < n_batches - 1:
                cp.vectorizer.start_session()
                if opts.report:
                    print('  * Started TF in {:6.2f}s'.format(time() - t_reset))

        if opts.report:
            mp, sp = divmod(time() - t_start, 60)
            print('  Completed doc batch: {:3d}/{}      '
                  '  Total time passed: {:3d}m{:0.2f}s\n'
                  ''.format(i+1, n_batches, int(mp), sp))

    # Merge
    # TODO: Title indexes
    t_merge = time()
    merged_ivfs = shard_date + '_all.ivfdata'
    merged_ivfs = p.join(opts.output_dir, merged_ivfs)
    merged_ivfs = check_unique(merged_ivfs)
    merged_index = shard_date + '_all.index'
    merged_index = p.join(opts.output_dir, merged_index)
    merged_index = check_unique(merged_index)
    if opts.report:
        print('\n  Merging {} on-disk'.format(merged_index.split('/')[-1]))

    n_vect = cp.index_builder.merge_IVFs(index_path=merged_index, ivfdata_path=merged_ivfs)

    if opts.report:
        mm, sm = divmod(time() - t_merge, 60)
        print('  Merged subindexes ({} vectors) in: {:3d}m{:0.2f}s'
              ''.format(n_vect, int(mm), sm))

    # Record progress
    cp.record_progress(raw_jl)

    # Clear sub.index files after merge
    if opts.delete_tmp_files:
        clear_dir(subidx_dir)
        if opts.report:
            print('\n  Cleared sub.index files')

    if opts.add_shard:
        try:
            url = opts.url
            payload = {'path': merged_index}
            r = requests.put(url, params=payload)
            print(r.text)
        except Exception as e:
            print('Shard was not added because an exception occurred: {}'.format(e))


if __name__ == '__main__':
    if len(file_to_process):
        main()
    else:
        print('Nothing to process.')
