# <editor-fold desc="Basic Imports">
import os
import os.path as p
from argparse import ArgumentParser

import sys
sys.path.append(p.join(p.dirname(__file__), '..'))
sys.path.append(p.join(p.dirname(__file__), '../..'))
# </editor-fold>

# <editor-fold desc="Parse Command Line Options">
arp = ArgumentParser(description='Use this script to move, copy, or zip/merge '
                                 'on-disk IVF faiss indexes from the command line. '
                                 'Note: Performing `$ mv my_IVF.index ...` from '
                                 'the command line will sever the link to its '
                                 'my_IVF.ivfdata file.')

arp.add_argument('mv_dir', help='Collect on-disk IVF indexes to manipulate from '
                                'mv_dir...')
arp.add_argument('to_dir', help='... place indexes (and corresponding '
                                '.ivfdata files) into to_dir.')

action = arp.add_mutually_exclusive_group(required=True)
action.add_argument('-z', '--zip', action='store_true', default=False,
                    help='Zipping automatically groups and merges indexes '
                         'that share an ISO-date in their filenames.')
action.add_argument('-m', '--mv', '--move', action='store_true', default=False,
                    help='Moves indexes from mv_dir to to_dir (does NOT zip '
                         'indexes together). Fails if existing filenames conflict! '
                         '(Note: may fail after partial completion)')
action.add_argument('-c', '--cp', '--copy', action='store_true', default=False,
                    help='Copies indexes from mv_dir to to_dir (does NOT delete '
                         'anything in mv_dir). Fails if existing filenames conflict! '
                         '(Note: may fail after partial completion)')

init = arp.add_argument_group()
base_dir_path = p.abspath(p.join(p.dirname(__file__), '../../base_indexes/'))
large_base_path = p.join(base_dir_path, 'USE_large_base_IVF4K_15M.index')
init.add_argument('-b', '--base_index', default=large_base_path,
                  help='This path must point to the same base index used for '
                       f'building the IVFs in mv_dir. (Default: {large_base_path})')
init.add_argument('-t', '--num_threads', default='2',
                  help='Set CPU thread budget for numpy.')

zipped = arp.add_argument_group()
zipped.add_argument('-p', '--PIN', '--partial-idx-name', default='zipped',
                    help='Will be used to rename zipped indexes '
                         '(i.e. YYYY-MM-DD_partial-idx-name.index). '
                         'Only applies for --zip. (Default: `zipped`)')
zipped.add_argument('-r', '--recursive', action='store_true', default=False,
                    help='Flag to recursively collect faiss indexes '
                         'nested under mv_dir. Only applies for --zip.')
opts = arp.parse_args()
# </editor-fold>

if opts.num_threads:
    print('\nRestricting numpy to {} thread(s)\n'.format(opts.num_threads))
    os.environ['OPENBLAS_NUM_THREADS'] = opts.num_threads
    os.environ['NUMEXPR_NUM_THREADS'] = opts.num_threads
    os.environ['MKL_NUM_THREADS'] = opts.num_threads
    os.environ['OMP_NUM_THREADS'] = opts.num_threads

from dt_sim.indexer.index_builder import OnDiskIVFBuilder


# Init
idx_bdr = OnDiskIVFBuilder(p.abspath(opts.base_index))


# Main
def main():
    if opts.mv or opts.cp:
        n_vect = idx_bdr.mv_indexes(
            mv_dir=p.abspath(opts.mv_dir), to_dir=p.abspath(opts.to_dir),
            mkdir=True, only_cp=opts.cp
        )
        print(f'\n * Final count: {len(idx_bdr.find_indexes(opts.to_dir))} indexes '
              f'with {n_vect} vectors in total \n')
    else:
        idx_bdr.zip_indexes(
            mv_dir=p.abspath(opts.mv_dir), to_dir=p.abspath(opts.to_dir),
            mkdir=True, partial_filename=opts.PIN, recursive=opts.recursive
        )


if __name__ == '__main__':
    main()
