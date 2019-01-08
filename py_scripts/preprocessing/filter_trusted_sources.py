# <editor-fold desc="Basic Imports">
import os.path as p
from argparse import ArgumentParser

import sys
sys.path.append(p.join(p.dirname(__file__), '..'))
sys.path.append(p.join(p.dirname(__file__), '../..'))
# </editor-fold>

from dt_sim.data_reader.source_filter_funcs import *

# <editor-fold desc="Parse Command Line Options">
arp = ArgumentParser(description='Append articles written by trusted sources '
                                 'from input_file.jl into output_file.jl')
arp.add_argument('input_file', help='Path to rawLexisNexis.jl to be filtered.')
arp.add_argument('output_file', help='Path to trustedLexisNexis.jl '
                                     '(appends new docs if file exists).')
arp.add_argument('-w', '--white_list', default=news_white_list,
                 help='Substitute your own news source white list '
                      '(default: dt_sim/data_reader/SourceWhiteList.txt)')
opts = arp.parse_args()
# </editor-fold>


if __name__ == '__main__':
    source_filter(
        input_file=opts.input_file,
        output_file=opts.output_file,
        white_list=opts.white_list
    )
