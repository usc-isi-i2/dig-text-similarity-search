# <editor-fold desc="Basic Imports">
import os
import os.path as p
import re
import sys
import json
import datetime
from time import time

sys.path.append(p.join(p.dirname(__file__), '..'))
sys.path.append(p.join(p.dirname(__file__), '../..'))
# </editor-fold>

# <editor-fold desc="Parse Command Line Options">
from optparse import OptionParser
arg_parser = OptionParser()
arg_parser.add_option('-i', '--input_file')
arg_parser.add_option('-o', '--output_dir')
arg_parser.add_option('-r', '--report', action='store_false', default=True)
arg_parser.add_option('-v', '--verbose_vectorizer', action='store_true', default=False)
arg_parser.add_option('-t', '--num_threads')
arg_parser.add_option('-l', '--USE_large', action='store_true', default=False)
arg_parser.add_option('-m', '--m_per_batch', type='int', default=512*128)
arg_parser.add_option('-n', '--n_per_minibatch', type='int', default=64)
arg_parser.add_option('-s', '--skip', type='int')
(opts, _) = arg_parser.parse_args()
# </editor-fold>

if opts.num_threads:
    print('\nRestricting numpy to {} thread(s)\n'.format(opts.num_threads))
    os.environ['OPENBLAS_NUM_THREADS'] = opts.num_threads
    os.environ['NUMEXPR_NUM_THREADS'] = opts.num_threads
    os.environ['MKL_NUM_THREADS'] = opts.num_threads
    os.environ['OMP_NUM_THREADS'] = opts.num_threads

import numpy as np

from dt_sim_api.data_reader.jl_io_funcs import check_training_docs, get_training_docs
from dt_sim_api.data_reader.npz_io_funcs import save_with_ids
from dt_sim_api.data_reader.misc_io_funcs import check_unique
from dt_sim_api.vectorizer.sentence_vectorizer import SentenceVectorizer
from dt_sim_api.processor.corpus_processor import CorpusProcessor


"""
Run this script to save embeddings for training a new base index. 

    Note: It may be necessary to write a new loading function, 
          depending on the corpus to be vectorized/indexed.

To run:
    $ python make_training_vectors.py -i /path/to/split_sentences.jl

Command Line Args:
    Required:
        -i  Path to file containing sentences to vectorize 
    
    Options:
        -o  Path to specify output dir
                (default: input_file_dir/../training_npzs/)
        -r  Toggle stdout prints 
                (bool: default True)
        -v  Toggle vectorizer performance stdout prints 
                (bool: default False)
        -t  Limit CPU thread budget for numpy/faiss 
                (int: defaults to all CPU threads)
        -l  Load large Universal Sentence Encoder [Transformer Network]
                (bool: defaults to USE-lite [Deep Averaging Network]) 
        -m  Minimum number of sentences per batch 
                Note: Actual batch size may vary (no document clipping)
                (int: default 512*128)
        -n  Size of vectorizer minibatch 
                (int: default 64)
    
        -s  Number of leading batches to skip
                Note: Useful if vectorization was interrupted after 
                      several training.npz files were saved
"""


# Init
sv = SentenceVectorizer(large=opts.USE_large)
cp = CorpusProcessor(vectorizer=sv)


def main():
    # Paths
    input_file = opts.input_file
    if opts.report:
        print('Will process: {}\n'.format(input_file))

    date_today = str(datetime.date.today())
    if date_today in input_file:
        date = date_today
    else:
        seed = str('\d{4}[-/]\d{2}[-/]\d{2}')
        date = re.search(seed, input_file).group()

    # Nested output dirs
    if not opts.output_dir:
        output_dir = p.abspath(p.join(p.dirname(input_file),
                                      '../training_npzs/'))
    else:
        output_dir = opts.output_dir
    daily_dir = p.join(output_dir, date)
    if not p.isdir(output_dir):
        os.mkdir(output_dir)
    if not p.isdir(daily_dir):
        os.mkdir(daily_dir)

    # Check File Content
    if opts.report:
        print('\nReading file: {}'.format(input_file))

    line_counts = check_training_docs(input_file, batch_size=opts.m_per_batch)
    (doc_count, line_count, good_sents, junk, n_batches, n_good_batches) = line_counts
    if opts.report:
        print('* Found {} good documents with {} lines and {} good sentences\n'
              '* Will skip {} junk documents\n'
              '* Processing {}:{} batches\n'
              ''.format(doc_count, line_count, good_sents,
                        junk, n_good_batches, n_batches))

    # Make Training Vectors
    t_start = time()
    doc_batch_gen = get_training_docs(input_file, batch_size=opts.m_per_batch)
    for i, (batched_sents, batched_ids) in enumerate(doc_batch_gen, start=1):
        t_0 = time()
        if opts.report:
            print('  Starting doc batch:  {:3d}'.format(i))

        npz = str(input_file.split('/')[-1]).replace('.jl', '_{:03d}_train.npz'.format(i))
        npz_path = p.join(daily_dir, npz)

        if opts.skip and i < opts.skip:
            print('  Skipping...  ')
        elif p.exists(npz_path):
            print('  File exists: {} \n'
                  '  Skipping...  '.format(npz_path))
        else:
            # Vectorize
            emb_batch, id_batch = cp.vectorize(text_batch=batched_sents,
                                               id_batch=batched_ids,
                                               n_minibatch=opts.n_per_minibatch,
                                               very_verbose=opts.verbose_vectorizer)
            t_vect = time()
            if opts.report:
                print('  * Vectorized in {:6.2f}s'.format(t_vect - t_0))

            # Save .npz for later
            npz_path = check_unique(npz_path)
            save_with_ids(npz_path, embeddings=emb_batch, sent_ids=id_batch,
                          sentences=batched_sents, compressed=False)
            t_npz = time()
            if opts.report:
                print('  * Saved .npz in {:6.2f}s'.format(t_npz - t_vect))

            # Clear graph
            del emb_batch, id_batch, batched_sents, batched_ids
            cp.vectorizer.close_session()
            t_reset = time()
            if opts.report:
                print('  * Cleared TF in {:6.2f}s'.format(t_reset - t_npz))

            # Restart TF session if necessary
            if i < n_batches:
                cp.vectorizer.start_session()
                if opts.report:
                    print('  * Started TF in {:6.2f}s'.format(time() - t_reset))

            if opts.report:
                mp, sp = divmod(time() - t_start, 60)
                print('  Completed doc batch: {:3d}/{}      '
                      '  Total time passed: {:3d}m{:0.2f}s\n'
                      ''.format(i, n_good_batches, int(mp), sp))


if __name__ == '__main__':
    if p.isfile(opts.input_file):
        main()
    else:
        print('File not found: {}'.format(opts.input_file))
