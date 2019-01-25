# <editor-fold desc="Basic Imports">
import os.path as p
from argparse import ArgumentParser

import sys
sys.path.append(p.join(p.dirname(__file__), '..'))
sys.path.append(p.join(p.dirname(__file__), '../..'))
# </editor-fold>

# <editor-fold desc="Parse Command Line Options">
arp = ArgumentParser(description='Use this script to move, copy, or zip/merge '
                                 'on-disk IVF faiss indexes from the command line. '
                                 'Zipping (default behavior) automatically groups '
                                 'and merges indexes that share an ISO-date in '
                                 'their filenames. '
                                 'Note: Performing `$ mv my_IVF.index ...` from '
                                 'the command line will sever the link to its '
                                 'my_IVF.ivfdata file.')

arp.add_argument('mv_dir', help='Collect on-disk IVF indexes to manipulate from '
                                'mv_dir...')
arp.add_argument('to_dir', help='... place indexes (and corresponding '
                                '.ivfdata files) into to_dir.')

base_idx = arp.add_mutually_exclusive_group()
base_dir_path = p.abspath(p.join(p.dirname(__file__), '../../base_indexes/'))
large_base_path = p.join(base_dir_path, 'USE_large_base_IVF4K_15M.index')
base_idx.add_argument('-b', '--base_index', default=large_base_path,
                      help='')

action = arp.add_mutually_exclusive_group()
action.add_argument('-r', '--recursive', action='store_true', default=False,
                    help='Flag to recursively collect faiss indexes '
                         'nested within mv_dir. (Only available for zip/merging)')
action.add_argument('-m', '--mv', '--move', action='store_true', default=False,
                    help='Moves indexes from mv_dir to to_dir (does NOT zip '
                         'indexes together). Fails if existing filenames conflict! '
                         '(Note: may fail after partial completion)')
action.add_argument('-c', '--cp', '--copy', action='store_true', default=False,
                    help='Copies indexes from mv_dir to to_dir (does NOT delete '
                         'anything in mv_dir). Fails if existing filenames conflict! '
                         '(Note: may fail after partial completion)')

arp.add_argument('-D', '--dont_mkdir', action='store_true', default=False,
                 help='Flag to prevent python from making to_dir if it '
                      'does not already exist.')
opts = arp.parse_args()
# </editor-fold>

from dt_sim.indexer.index_builder import OnDiskIVFBuilder


# Init
idx_bdr = OnDiskIVFBuilder(p.abspath(opts.base_index))


# Main
def main():
    if opts.mv or opts.cp:
        idx_bdr.mv_indexes(
            mv_dir=p.abspath(opts.mv_dir), to_dir=p.abspath(opts.to_dir),
            mkdir=not opts.dont_mkdir, only_cp=opts.cp
        )
    else:
        idx_bdr.zip_indexes(
            mv_dir=p.abspath(opts.mv_dir), to_dir=p.abspath(opts.to_dir),
            mkdir=not opts.dont_mkdir, recursive=opts.recursive
        )
