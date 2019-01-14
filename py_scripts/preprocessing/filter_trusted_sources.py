# <editor-fold desc="Basic Imports">
import os.path as p
from argparse import ArgumentParser

import sys
sys.path.append(p.join(p.dirname(__file__), '..'))
sys.path.append(p.join(p.dirname(__file__), '../..'))
# </editor-fold>

# <editor-fold desc="Parse Command Line Options">
arp = ArgumentParser(description='Append articles written by trusted sources '
                                 'from input_file.jl into output_file.jl')
arp.add_argument('input_file', help='Path to rawLexisNexis.jl to be filtered.')
arp.add_argument('output_file', help='Path to trustedLexisNexis.jl '
                                     '(appends new docs if file exists).')
arp.add_argument('-w', '--white_list_path',
                 default='py_scripts/configs/SourceWhiteList.txt',
                 help='Substitute your own news source white list '
                      '(default: py_scripts/configs/SourceWhiteList.txt)')
opts = arp.parse_args()
# </editor-fold>

from dt_sim.data_reader.source_filter_funcs import source_filter


wl_file = p.abspath(opts.white_list_path)
try:
    news_white_list = list()
    with open(wl_file, 'r') as wl:
        for ln in wl:
            news_white_list.append(str(ln).replace('\n', ''))
    news_white_list = tuple(news_white_list)
except FileNotFoundError:
    news_white_list = None
    print(f'File not found: {wl_file}')


if __name__ == '__main__' and news_white_list:
    source_filter(
        input_file=opts.input_file,
        output_file=opts.output_file,
        white_list=news_white_list
    )
