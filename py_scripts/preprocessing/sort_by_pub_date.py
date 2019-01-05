# <editor-fold desc="Basic Imports">
import os.path as p
from argparse import ArgumentParser

import sys
sys.path.append(p.join(p.dirname(__file__), '..'))
sys.path.append(p.join(p.dirname(__file__), '../..'))
# </editor-fold>

# <editor-fold desc="Parse Command Line Options">
arp = ArgumentParser(description='Split a LexisNexis news_dump.jl '
                                 'by article publication dates.')
arp.add_argument('input_file', help='Path to dirtyLexisNexis.jl to be sorted.')
arp.add_argument('output_dir', help='Dir for saving cleanLexisNexis.jl files.')
arp.add_argument('-c', '--cutoff_date', default='2018-01-01',
                 help='Articles published after the cutoff date will be '
                      'saved in output_dir/old_news/*.jl separately.')
opts = arp.parse_args()
# </editor-fold>

from dt_sim.data_reader.date_sort_funcs import pub_date_split


if __name__ == '__main__':
    pub_date_split(
        input_file=opts.input_file,
        output_dir=opts.output_dir,
        cutoff_date=opts.cutoff_date
    )
