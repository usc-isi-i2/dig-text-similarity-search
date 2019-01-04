# <editor-fold desc="Basic Imports">
import os.path as p

import sys
sys.path.append(p.join(p.dirname(__file__), '..'))
sys.path.append(p.join(p.dirname(__file__), '../..'))
# </editor-fold>

# <editor-fold desc="Parse Command Line Options">
from argparse import ArgumentParser
parser = ArgumentParser(description='Split a LexisNexis news_dump.jl '
                                    'by article publication dates.')
parser.add_argument('input_file', help='Path to dirtyLexisNexis.jl to be sorted.')
parser.add_argument('output_dir', help='Dir for saving cleanLexisNexis.jl files.')
parser.add_argument('-c', '--cutoff_date', default='2018-01-01',
                    help='Articles published after the cutoff date will be '
                         'saved in output_dir/old_news/*.jl separately.')
args = parser.parse_args()
# </editor-fold>

from dt_sim.data_reader.date_sort_funcs import pub_date_split


if __name__ == '__main__':
    pub_date_split(
        input_file=args.input_file,
        output_dir=args.output_dir,
        cutoff_date=args.cutoff_date
    )
