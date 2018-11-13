# <editor-fold desc="Imports">
import os
from optparse import OptionParser
# <editor-fold desc="Parse Command Line Options">

options = OptionParser()
options.add_option('-i', '--input_file')
options.add_option('-t', '--num_threads')
options.add_option('-a', '--intra', type='int', default=8)  # TODO: fix TF Config
options.add_option('-e', '--inter', type='int', default=2)
options.add_option('-m', '--m_per_batch', type='int', default=512*128)
options.add_option('-n', '--n_per_minibatch', type='int', default=128)
options.add_option('-l', '--USE_large', action='store_true', default=False)
options.add_option('-r', '--report', action='store_true', default=False)
options.add_option('-v', '--verbose', action='store_true', default=False)
options.add_option('-s', '--skip', type='int', default=0)
(opts, _) = options.parse_args()
# </editor-fold>

if opts.num_threads:
    print('\nRestricting numpy to {} thread(s)\n'.format(opts.num_threads))
    os.environ['OPENBLAS_NUM_THREADS'] = opts.num_threads
    os.environ['NUMEXPR_NUM_THREADS'] = opts.num_threads
    os.environ['MKL_NUM_THREADS'] = opts.num_threads
    os.environ['OMP_NUM_THREADS'] = opts.num_threads

import re
import sys
import json
import datetime
import numpy as np
from time import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from dt_sim_api.data_reader.io_funcs \
    import check_training_docs, aggregate_training_docs, check_unique
from dt_sim_api.vectorizer.sentence_vectorizer import SentenceVectorizer
from dt_sim_api.process_documents.document_processor import DocumentProcessor
# </editor-fold>


"""
To run this script:
    (from dig-text-similarity-search/)
    $ python scripts/preprocessing/training_npzs.py \ 
        -i {/path/to/split_sents.jl} -r (Optional: -l)

Options:
    -i  Path to raw news dir
    -t  Option to set thread budget for numpy to reduce CPU resource consumption 
            Useful if other tasks are running 
    -a  
    -e  
    -m  Minimum number of sentences per batch 
            (default 512*128)
    -n  
    -l  Bool to use large Universal Sentence Encoder
            (default False) 
    -r  Bool to toggle prints 
            (default False)
    -v  

    -s  Development param: If preprocessing was interrupted after several 
            sub.index files were created, but before the on-disk shard was merged, 
            use -s <int:n_files_to_reuse> to reuse existing intermediate files. 
            * Note: Do NOT reuse partially created intermediate files
"""


# Track progress
file_to_process = [opts.input_file]
if opts.report:
    print('Will process: {}\n'.format(file_to_process[0]))


# Init paths
date_today = str(datetime.date.today())
if date_today in file_to_process[0]:
    date = date_today
else:
    seed = str('\d{4}[-/]\d{2}[-/]\d{2}')
    date = re.search(seed, file_to_process[0]).group()

intermediate_dir = os.path.abspath(os.path.join(os.path.dirname(opts.input_file), 
                                                '../training_npzs/'))
daily_dir = os.path.join(intermediate_dir, date)
if not os.path.isdir(intermediate_dir):
    os.mkdir(intermediate_dir)
if not os.path.isdir(daily_dir):
    os.mkdir(daily_dir)

if opts.USE_large:
    large_dir = '../digtextsimilaritysearch/vectorizer' \
                '/model/96e8f1d3d4d90ce86b2db128249eb8143a91db73'
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), large_dir))
else:
    model_dir = None


sv = SentenceVectorizer(path_to_model=model_dir, intra=opts.intra, inter=opts.inter)
dp = DocumentProcessor(indexer=None, index_builder=None,
                       vectorizer=sv, storage_adapter=None)


# Preprocessing
def main():
    for raw_jl in file_to_process:
        if opts.report:
            print('\nReading file: {}'.format(raw_jl))

        line_counts = check_training_docs(file_path=raw_jl, b_size=opts.m_per_batch)
        (doc_count, line_count, good_sents, junk, n_batches, n_good_batches) = line_counts
        if opts.report:
            print('* Found {} good documents with {} lines and {} good sentences\n'
                  '* Will skip {} junk documents\n'
                  '* Processing {}:{} batches\n'
                  ''.format(doc_count, line_count, good_sents,
                            junk, n_good_batches, n_batches))

        t_start = time()
        doc_batch_gen = aggregate_training_docs(file_path=raw_jl, b_size=opts.m_per_batch)
        for i, (batched_sents, batched_ids) in enumerate(doc_batch_gen):
            t_0 = time()
            if opts.report:
                print('  Starting doc batch:  {:3d}'.format(i+1))

            npz = str(raw_jl.split('/')[-1]).replace('.jl', '_{:03d}_train.npz'.format(i))
            npz_path = os.path.join(daily_dir, npz)

            # TODO add skip
            if i < opts.skip and os.path.exists(npz_path):
                print('  File exists: {} \n'
                      '  Skipping'.format(npz_path))
            else:
                # Vectorize
                batched_embs = dp.vectorizer.make_vectors(batched_sents,
                                                          batch_size=opts.n_per_minibatch,
                                                          verbose=opts.verbose)
                t_vect = time()
                if opts.report:
                    print('  * Vectorized in {:6.2f}s'.format(t_vect - t_0))

                # Numpify
                if not isinstance(batched_embs, np.ndarray):
                    batched_embs = np.vstack(batched_embs).astype(np.float32)
                if not isinstance(batched_ids, np.ndarray):
                    try:
                        batched_ids = np.array(batched_ids, dtype=np.int64)
                    except ValueError:
                        print(batched_ids)
                        raise ValueError

                # Save npz for later
                npz_path = check_unique(path=npz_path)
                dp.vectorizer.save_with_ids(npz_path,
                                            embeddings=batched_embs,
                                            sentences=batched_sents,
                                            sent_ids=batched_ids,
                                            compressed=False)

                t_npz = time()
                if opts.report:
                    print('  * Saved .npz in {:6.2f}s'.format(t_npz - t_vect))

                # Clear graph
                del batched_embs, batched_sents, batched_ids
                dp.vectorizer.close_session()
                t_reset = time()
                if opts.report:
                    print('  * Cleared TF in {:6.2f}s'.format(t_reset - t_npz))

                # Restart TF session if necessary
                if i < n_batches - 1:
                    dp.vectorizer.start_session()
                    if opts.report:
                        print('  * Started TF in {:6.2f}s'.format(time() - t_reset))

                if opts.report:
                    mp, sp = divmod(time() - t_start, 60)
                    print('  Completed doc batch: {:3d}/{}      '
                          '  Total time passed: {:3d}m{:0.2f}s\n'
                          ''.format(i+1, n_good_batches, int(mp), sp))


if __name__ == '__main__':
    if len(file_to_process):
        main()
    else:
        print('Nothing to process.')
