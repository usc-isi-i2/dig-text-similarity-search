# <editor-fold desc="Basic Imports">
import os
import os.path as p
from time import time
from optparse import OptionParser

import sys
sys.path.append(p.join(p.dirname(__file__), '..'))
sys.path.append(p.join(p.dirname(__file__), '../..'))
# </editor-fold>

# <editor-fold desc="Parse Options">
arg_parser = OptionParser()
arg_parser.add_option('-i', '--input_npz_dir')
arg_parser.add_option('-o', '--output_index_dir')
arg_parser.add_option('-b', '--base_index_name', default='emptyTrainedIVF.index')
arg_parser.add_option('-m', '--mmap_name')
arg_parser.add_option('-n', '--n_centroids', default='4096')
arg_parser.add_option('-c', '--compression', default='Flat')
arg_parser.add_option('-v', '--verbose', action='store_true', default=False)
arg_parser.add_option('-t', '--num_threads')
arg_parser.add_option('-N', '--N_training_vectors', type='int', default=1000000)
(opts, _) = arg_parser.parse_args()
# </editor-fold>

# <editor-fold desc="Limit Numpy Threads">
if opts.num_threads:
    print('\nRestricting numpy to {} thread(s)\n'.format(opts.num_threads))
    os.environ['OPENBLAS_NUM_THREADS'] = opts.num_threads
    os.environ['NUMEXPR_NUM_THREADS'] = opts.num_threads
    os.environ['MKL_NUM_THREADS'] = opts.num_threads
    os.environ['OMP_NUM_THREADS'] = opts.num_threads
# </editor-fold>

from dt_sim_api.indexer.index_builder import LargeIndexBuilder


"""
Script for training a base IVF index for later use.

Options:
    -i  Full path to .npz directory
    -o  Full path to index directory
    -b  Name of empty index to be trained
    -n  Number of .npz files to train on
"""

# Set up paths
if not p.isdir(opts.output_index_dir):
    os.mkdir(opts.output_index_dir)
base_index_path = p.abspath(p.join(opts.output_index_dir, opts.base_index_name))
training_set_path = p.abspath(p.join(opts.input_npz_dir, opts.mmap_name))

# Init
idx_bdr = LargeIndexBuilder(path_to_base_index=base_index_path)


# Main
def main():
    t_start = time()

    idx_bdr.setup_base_index(
        base_index_path=base_index_path,
        centroids=opts.n_centroids, ts_path=training_set_path,
        npz_dir=opts.input_npz_dir, n_tr_vectors=opts.N_training_vectors
    )

    print('\nProcess completed in {:0.2f}s'.format(time()-t_start))


if __name__ == '__main__':
    main()
