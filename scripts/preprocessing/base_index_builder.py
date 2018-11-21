# <editor-fold desc="Imports">
import os
from time import time
from optparse import OptionParser
# <editor-fold desc="Parse Options">
arg_parser = OptionParser()
arg_parser.add_option('-i', '--input_npz_dir')
arg_parser.add_option('-o', '--output_dir')
arg_parser.add_option('-b', '--base_index_name', default='emptyTrainedIVF.index')
arg_parser.add_option('-m', '--mmap_name')
arg_parser.add_option('-n', '--n_centroids', default='4096')
arg_parser.add_option('-c', '--compression', default='Flat')
arg_parser.add_option('-v', '--verbose', action='store_true', default=False)
arg_parser.add_option('-t', '--num_threads')
arg_parser.add_option('-N', '--N_training_vectors', type='int', default=1000000)
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

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from dt_sim_api.data_reader.npz_io_funcs import *
# </editor-fold>


"""
Script for training a base IVF index for later use.

Options:
    -i  Full path to .npz directory
    -o  Full path to index directory
    -b  Name of empty index to be trained
    -n  Number of .npz files to train on
"""


def make_base_IVF(training_set, save_path, centroids,
                  compression: str = 'Flat', dim: int = 512):
    # Create base IVF index
    index_type = 'IVF{},{}'.format(centroids, compression)
    print('\nCreating base faiss index: {}'.format(index_type))
    index = faiss.index_factory(dim, index_type)

    # Train
    print(' Training...')
    t_train0 = time()
    index.train(training_set)
    t_train1 = time()
    print(' Index trained in {:0.2f}s'.format(t_train1-t_train0))

    # Save
    print(' Saving trained base index...')
    faiss.write_index(index, save_path)
    print(' Index saved in {:0.2f}s'.format(time()-t_train1))


# Main
def main():
    t_start = time()

    # Set up paths
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    base_index_path = os.path.abspath(os.path.join(args.output_dir,
                                                   args.base_index_name))

    # Load
    training_set = load_training_npz(args.input_npz_dir,
                                     training_set_name=args.mmap_name,
                                     n_vectors=args.N_training_vectors)

    # Train
    make_base_IVF(training_set, save_path=base_index_path,
                  centroids=args.n_centroids, compression=args.compression)

    print('\nProcess completed in {:0.2f}s'.format(time()-t_start))


if __name__ == '__main__':
    main()
