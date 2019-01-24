import json
from time import time
from typing import Union, List, Tuple

__all__ = ['source_filter']


def source_filter(input_file, output_file,
                  white_list: Union[List, Tuple] = None):
    t_0 = time()
    n_files, m_good = 0, 0
    with open(input_file, 'r') as src, open(output_file, 'a') as dst:
        for line in src:
            n_files += 1
            doc = json.loads(line)
            if doc['lexisnexis']['metadata']['source'] in white_list:
                m_good += 1
                dst.write(json.dumps(doc) + '\n')

    m, s = divmod(time() - t_0, 60)
    print(f'{n_files} files sorted in {int(m):2d}m{s:0.1f}s '
          f'({100*m_good/n_files:0.1f}% from trusted sources)')
