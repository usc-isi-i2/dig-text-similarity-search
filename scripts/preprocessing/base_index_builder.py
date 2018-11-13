# <editor-fold desc="Imports">
import os
import sys
from optparse import OptionParser
# <editor-fold desc="Parse Options">
arg_parser = OptionParser()
arg_parser.add_option('-i', '--input_npz_dir')
arg_parser.add_option('-o', '--output_dir')
arg_parser.add_option('-m', '--mmap_name')
arg_parser.add_option('-n', '--n_centroids', default='4096')
arg_parser.add_option('-c', '--compression', default='Flat')
arg_parser.add_option('-b', '--base_index_name', default='emptyTrainedIVF.index')
arg_parser.add_option('-v', '--verbose', action='store_true', default=False)
arg_parser.add_option('-t', '--num_threads')
(args, _) = arg_parser.parse_args()
# </editor-fold>

# <editor-fold desc="Limit Numpy Threads">
if args.num_threads:
    print('\nRestricting numpy to {} thread(s)\n'.format(opts.num_threads))
    os.environ['OPENBLAS_NUM_THREADS'] = opts.num_threads
    os.environ['NUMEXPR_NUM_THREADS'] = opts.num_threads
    os.environ['MKL_NUM_THREADS'] = opts.num_threads
    os.environ['OMP_NUM_THREADS'] = opts.num_threads
# </editor-fold>

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import faiss
import numpy as np
from time import time
from dt_sim_api.vectorizer import SentenceVectorizer
# </editor-fold>


"""
Script for training a base IVF index for later use.

Options:
    -i  Full path to .npz directory
    -o  Full path to index directory
    -b  Name of empty index to be trained
    -n  Number of .npz files to train on
"""


# Init
sv = SentenceVectorizer()


# Funcs
def get_all_npz_paths(npz_parent_dir):
    npz_paths = list()
    for (dirpath, _, filenames) in os.walk(npz_parent_dir, topdown=True):
        for f in filenames:
            if f.endswith('.npz'):
                npz_paths.append(os.path.join(dirpath, f))
    return sorted(npz_paths)


def load_training_npz(npz_paths, tmp_name, sentence_vectorizer=sv, mmap=True):
    t_load = time()

    emb_list = list()
    emb_lens = list()
    for npzp in npz_paths:
        emb, _, _ = sentence_vectorizer.load_with_ids(npzp, mmap=mmap)
        emb_list.append(emb), emb_lens.append(emb.shape)

    tot_embs = sum([n[0] for n in emb_lens])
    emb_wide = emb_lens[0][1]
    print('\nFound {} vectors of {}d'.format(tot_embs, emb_wide))

    ts_memmap = np.memmap(tmp_name, dtype=np.float32,
                          mode='w+', shape=(tot_embs, emb_wide))

    place = 0
    for emb in emb_list:
        n_vect = emb.shape[0]
        ts_memmap[place:place+n_vect, :] = emb[:]
        place += n_vect

    m, s = divmod(time()-t_load, 60)
    print(' Training set loaded in {}m{:0.2f}s'.format(int(m), s))

    return ts_memmap


def make_base_IVF(training_set, save_path, centroids, compression):
    # Create base IVF index
    index_type = 'IVF{},{}'.format(centroids, compression)
    print('\nCreating index: {}'.format(index_type))
    index = faiss.index_factory(training_set.shape[1], index_type)

    # Train
    print(' Training index...')
    t_train0 = time()
    index.train(training_set)
    t_train1 = time()
    print(' Index trained in {:0.2f}s'.format(t_train1-t_train0))

    # Save
    print(' Saving index...')
    faiss.write_index(index, save_path)
    print(' Index saved in {:0.2f}s'.format(time()-t_train1))


# Main
def main():
    t_start = time()

    # Set up paths
    assert os.path.isdir(args.input_npz_dir), \
        'Input npz dir path does not exist: {}'.format(args.input_npz_dir)
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    base_index_path = os.path.join(args.output_dir, args.base_index_name)

    # Gather
    all_npz_paths = get_all_npz_paths(args.input_npz_dir)

    # Load
    training_set = load_training_npz(all_npz_paths, tmp_name=args.mmap_name,
                                     sentence_vectorizer=sv, mmap=True)

    # Train
    make_base_IVF(training_set, save_path=base_index_path,
                  centroids=args.n_centroids, compression=args.compression)

    print('\nProcess completed in {:0.2f}s'.format(time()-t_start))


if __name__ == '__main__':
    main()
