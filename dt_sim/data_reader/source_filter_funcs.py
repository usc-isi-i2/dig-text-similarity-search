import json
import os.path as p
from typing import Union, List, Tuple


__all__ = ['news_white_list', 'source_filter']


wl_file = p.abspath(p.join(p.dirname(__name__), 'SourceWhiteList.txt'))
try:
    news_white_list = list()
    with open(wl_file, 'r') as wl:
        for ln in wl:
            news_white_list.append(str(ln).replace('\n', ''))
    news_white_list = tuple(news_white_list)
except FileNotFoundError:
    news_white_list = None


def source_filter(input_file, output_file,
                  white_list: Union[List, Tuple] = news_white_list):
    n_files, m_good = 0, 0
    with open(input_file, 'r') as src, open(output_file, 'a') as dst:
        for line in src:
            n_files += 1
            doc = json.loads(line)
            if doc['lexisnexis']['metadata']['source'] in white_list:
                m_good += 1
                dst.write(json.dumps(doc) + '\n')
    print(f'{n_files} files sorted '
          f'({100*m_good/n_files:0.1f}% from trusted sources)')
