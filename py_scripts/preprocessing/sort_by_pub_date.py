# <editor-fold desc="Basic Imports">
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
# </editor-fold>

# <editor-fold desc="Parse Command Line Options">
from argparse import ArgumentParser
parser = ArgumentParser(description='Split a LexisNexis news_dump.jl '
                                    'by article publication dates.')
parser.add_argument('-i', '--input_file', dest='input_file')
parser.add_argument('-o', '--output_dir', dest='output_dir')
parser.add_argument('-c', '--cutoff_date', dest='cutoff_date',
                    default='2018-01-01')
args = parser.parse_args()
# </editor-fold>

from dt_sim.data_reader.date_sort_funcs import pub_date_split


if __name__ == '__main__':
    pub_date_split(
        input_file=args.input_file,
        output_dir=args.output_dir,
        cutoff_date=args.cutoff_date
    )
