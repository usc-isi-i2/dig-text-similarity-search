# <editor-fold desc="Basic Imports">
import os
import os.path as p
import re
import sys
import datetime
from time import time
from argparse import ArgumentParser

sys.path.append(p.join(p.dirname(__file__), '..'))
sys.path.append(p.join(p.dirname(__file__), '../..'))
# </editor-fold>

# <editor-fold desc="Parse Command Line Options">
arp = ArgumentParser(description='Prepare vectors for training a base index.')

arp.add_argument('input_file', help='')
arp.add_argument('output_dir', help='')
arp.add_argument('-v', '--verbose', action='store_true',
                 help='Shows progress of batch vectorization.')
arp.add_argument('-V', '--verbose_vectorizer', action='store_true', default=False,
                 help='Shows vectorization time for each mini-batch.')
arp.add_argument('-l', '--USE_large', action='store_true',
                 help='Toggle large Universal Sentence Encoder (Transformer NN).')
arp.add_argument('-m', '--m_per_batch', type=int, default=512*128,
                 help='Sentences per batch.')
arp.add_argument('-n', '--n_per_minibatch', type=int, default=64,
                 help='Sentences per mini-batch.')
arp.add_argument('-t', '--num_threads',
                 help='Set CPU thread budget for numpy.')
opts = arp.parse_args()
# </editor-fold>

if opts.num_threads:
    print(f'\nRestricting numpy to {opts.num_threads} thread(s)\n')
    os.environ['OPENBLAS_NUM_THREADS'] = opts.num_threads
    os.environ['NUMEXPR_NUM_THREADS'] = opts.num_threads
    os.environ['MKL_NUM_THREADS'] = opts.num_threads
    os.environ['OMP_NUM_THREADS'] = opts.num_threads

from dt_sim.data_reader.jl_io_funcs import check_training_docs, get_training_docs
from dt_sim.data_reader.npz_io_funcs import save_with_ids
from dt_sim.data_reader.misc_io_funcs import check_unique
from dt_sim.vectorizer.sentence_vectorizer import SentenceVectorizer
from dt_sim.processor.corpus_processor import CorpusProcessor


# Init
sv = SentenceVectorizer(large=opts.USE_large)
cp = CorpusProcessor(vectorizer=sv)


def main():
    # Paths
    input_file = opts.input_file
    if opts.report:
        print(f'Will process: {input_file}\n')

    date_today = str(datetime.date.today())
    if date_today in input_file:
        date = date_today
    else:
        seed = str(r'\d{4}[-/]\d{2}[-/]\d{2}')
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
        print(f'\nReading file: {input_file}')

    line_counts = check_training_docs(input_file, batch_size=opts.m_per_batch)
    (doc_count, line_count, good_sents, junk, n_batches, n_good_batches) = line_counts
    if opts.report:
        print(f'* Found {doc_count} good documents with {line_count} lines '
              f'and {good_sents} good sentences\n'
              f'* Will skip {junk} junk documents\n'
              f'* Processing {n_good_batches}:{n_batches} batches\n')

    # Make Training Vectors
    t_start = time()
    doc_batch_gen = get_training_docs(input_file, batch_size=opts.m_per_batch)
    for i, (batched_sents, batched_ids) in enumerate(doc_batch_gen, start=1):
        t_0 = time()
        if opts.report:
            print(f'  Starting doc batch:  {i:3d}')

        npz = str(input_file.split('/')[-1]).replace('.jl', f'_{i:03d}_train.npz')
        npz_path = p.join(daily_dir, npz)

        if p.exists(npz_path):
            print(f'  File exists: {npz_path} \n Skipping...  ')
        else:
            # Vectorize
            emb_batch, id_batch = cp.batch_vectorize(
                text_batch=batched_sents, id_batch=batched_ids,
                n_minibatch=opts.n_per_minibatch, very_verbose=opts.verbose_vectorizer
            )
            t_vect = time()
            if opts.report:
                print(f'  * Vectorized in {t_vect - t_0:6.2f}s')

            # Save .npz for later
            npz_path = check_unique(npz_path)
            save_with_ids(npz_path, embeddings=emb_batch, sent_ids=id_batch,
                          sentences=batched_sents, compressed=False)
            t_npz = time()
            if opts.report:
                print(f'  * Saved .npz in {t_npz - t_vect:6.2f}s')

            # Clear graph
            del emb_batch, id_batch, batched_sents, batched_ids
            cp.vectorizer.close_session()
            t_reset = time()
            if opts.report:
                print(f'  * Cleared TF in {t_reset - t_npz:6.2f}s')

            # Restart TF session if necessary
            if i < n_batches:
                cp.vectorizer.start_session()
                if opts.report:
                    print(f'  * Started TF in {time() - t_reset:6.2f}s')

            if opts.report:
                mp, sp = divmod(time() - t_start, 60)
                print(f'  Completed doc batch: {i:3d}/{n_good_batches}      '
                      f'  Total time passed: {int(mp):3d}m{sp:0.2f}s\n')


if __name__ == '__main__':
    if p.isfile(opts.input_file):
        main()
    else:
        print(f'File not found: {opts.input_file}')
