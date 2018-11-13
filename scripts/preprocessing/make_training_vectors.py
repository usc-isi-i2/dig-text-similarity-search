# <editor-fold desc="Imports">
import os
import re
import json
import datetime
from time import time
from optparse import OptionParser

# <editor-fold desc="Parse Command Line Options">
options = OptionParser()

# Required
options.add_option('-i', '--input_file')

# Prints
options.add_option('-r', '--report', action='store_false', default=True)
options.add_option('-v', '--verbose_vectorizer', action='store_true', default=False)

# Faiss
options.add_option('-t', '--num_threads')

# Tensorflow
options.add_option('-l', '--USE_large', action='store_true', default=False)
options.add_option('-m', '--m_per_batch', type='int', default=512*128)
options.add_option('-n', '--n_per_minibatch', type='int', default=128)
options.add_option('-a', '--intra', type='int', default=8)  # TODO: fix TF Config
options.add_option('-e', '--inter', type='int', default=2)

# Dev
options.add_option('-s', '--skip', type='int')

(opts, _) = options.parse_args()
# </editor-fold>

if opts.num_threads:
    print('\nRestricting numpy to {} thread(s)\n'.format(opts.num_threads))
    os.environ['OPENBLAS_NUM_THREADS'] = opts.num_threads
    os.environ['NUMEXPR_NUM_THREADS'] = opts.num_threads
    os.environ['MKL_NUM_THREADS'] = opts.num_threads
    os.environ['OMP_NUM_THREADS'] = opts.num_threads

import numpy as np

from dt_sim_api.data_reader.io_funcs \
    import check_training_docs, aggregate_training_docs, check_unique
from dt_sim_api.vectorizer.sentence_vectorizer import SentenceVectorizer
from dt_sim_api.process_documents.document_processor import DocumentProcessor
# </editor-fold>


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
                Note:   128 :: USE-lite :::: 16 :: USE-large
                (int: default 128)
        # TODO: finalize TF.Config usage
        -a  Intra op parallelism threads for TF.Session(config)
                (int: default 8)
        -e  Inter op parallelism threads for TF.Session(config)
                (int: default 2)
    
        -s  Number of leading batches to skip
                Note: Useful if vectorization was interrupted after 
                      several training.npz files were saved
"""


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

intermediate_dir = os.path.abspath(os.path.join(os.path.dirname(opts.input_file), 
                                                '../training_npzs/'))
daily_dir = os.path.join(intermediate_dir, date)
if not os.path.isdir(intermediate_dir):
    os.mkdir(intermediate_dir)
if not os.path.isdir(daily_dir):
    os.mkdir(daily_dir)

if opts.USE_large:
    large_dir = '../../dt_sim_api/vectorizer/model/' \
                '96e8f1d3d4d90ce86b2db128249eb8143a91db73'
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), large_dir))
else:
    model_dir = None


# Init
sv = SentenceVectorizer(path_to_model=model_dir,
                        intra=opts.intra, inter=opts.inter)
dp = DocumentProcessor(indexer=None, index_builder=None,
                       vectorizer=sv, storage_adapter=None)


# Preprocessing
def main():
    if opts.report:
        print('\nReading file: {}'.format(input_file))

    line_counts = check_training_docs(file_path=input_file, b_size=opts.m_per_batch)
    (doc_count, line_count, good_sents, junk, n_batches, n_good_batches) = line_counts
    if opts.report:
        print('* Found {} good documents with {} lines and {} good sentences\n'
              '* Will skip {} junk documents\n'
              '* Processing {}:{} batches\n'
              ''.format(doc_count, line_count, good_sents,
                        junk, n_good_batches, n_batches))

    t_start = time()
    doc_batch_gen = aggregate_training_docs(file_path=input_file, b_size=opts.m_per_batch)
    for i, (batched_sents, batched_ids) in enumerate(doc_batch_gen, start=1):
        t_0 = time()
        if opts.report:
            print('  Starting doc batch:  {:3d}'.format(i))

        npz = str(input_file.split('/')[-1]).replace('.jl', '_{:03d}_train.npz'.format(i))
        npz_path = os.path.join(daily_dir, npz)

        if opts.skip and i <= opts.skip and os.path.exists(npz_path):
            print('  File exists: {} \n'
                  '  Skipping...  '.format(npz_path))
        else:
            # Vectorize
            batched_embs = dp.vectorizer.make_vectors(batched_sents,
                                                      batch_size=opts.n_per_minibatch,
                                                      verbose=opts.verbose_vectorizer)
            t_vect = time()
            if opts.report:
                print('  * Vectorized in {:6.2f}s'.format(t_vect - t_0))

            # Numpify
            if not isinstance(batched_embs, np.ndarray):
                batched_embs = np.vstack(batched_embs).astype(np.float32)
            if not isinstance(batched_ids, np.ndarray):
                try:
                    batched_ids = np.array(batched_ids, dtype=np.int64)
                except ValueError as e:
                    print('Cannot np.vstack: \n{}\n'.format(batched_ids))
                    print(e)

            # Save .npz for later
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
            if i < n_batches:
                dp.vectorizer.start_session()
                if opts.report:
                    print('  * Started TF in {:6.2f}s'.format(time() - t_reset))

            if opts.report:
                mp, sp = divmod(time() - t_start, 60)
                print('  Completed doc batch: {:3d}/{}      '
                      '  Total time passed: {:3d}m{:0.2f}s\n'
                      ''.format(i, n_good_batches, int(mp), sp))


if __name__ == '__main__':
    if os.path.isfile(input_file):
        main()
    else:
        print('File not found: {}'.format(input_file))
