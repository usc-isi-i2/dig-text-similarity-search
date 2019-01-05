# <editor-fold desc="Basic Imports">
import os
import os.path as p
from time import time
from argparse import ArgumentParser

import sys
sys.path.append(p.join(p.dirname(__file__), '..'))
sys.path.append(p.join(p.dirname(__file__), '../..'))
# </editor-fold>

# <editor-fold desc="Parse Options">
arp = ArgumentParser(description='Train a base IVF index for later use.')

arp.add_argument('input_npz_dir', help='Path to .npz directory.')
arp.add_argument('output_index_dir', help='Path to index directory.')
arp.add_argument('mmap_name', help='Filename for mmap training data.')
arp.add_argument('-b', '--base_index_name', default='emptyTrainedIVF.index',
                 help='Name of empty index to be trained.')
arp.add_argument('-m', '--m_training_vectors', type=int, default=1000000, 
                 help='Number of vectors to train on.')
arp.add_argument('-n', '--n_centroids', default='4096', 
                 help='Number of centroids in base IVF index.')
arp.add_argument('-c', '--compression', default='Flat', 
                 help='For faiss index constructor.')
arp.add_argument('-t', '--num_threads',
                 help='Set CPU thread budget for numpy.')
opts = arp.parse_args()
# </editor-fold>

# <editor-fold desc="Limit Numpy Threads">
if opts.num_threads:
    print('\nRestricting numpy to {} thread(s)\n'.format(opts.num_threads))
    os.environ['OPENBLAS_NUM_THREADS'] = opts.num_threads
    os.environ['NUMEXPR_NUM_THREADS'] = opts.num_threads
    os.environ['MKL_NUM_THREADS'] = opts.num_threads
    os.environ['OMP_NUM_THREADS'] = opts.num_threads
# </editor-fold>

from dt_sim.indexer.index_builder import LargeIndexBuilder


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
        npz_dir=opts.input_npz_dir, n_tr_vectors=opts.m_training_vectors
    )

    print('\nProcess completed in {:0.2f}s'.format(time()-t_start))


if __name__ == '__main__':
    main()
