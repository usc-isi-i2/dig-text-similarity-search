import os
import os.path as p
import json
from time import time

__all__ = ['pub_date_split']


def pub_date_split(input_file: str, output_dir: str,
                   cutoff_date: str = '2018-01-01'):
    """
    Sorts LexisNexis articles by publication date.
    :param input_file: /path/to/LexisNexisCrawlerDump.jl
    :param output_dir: Output destination of pub_date.jl files
    :param cutoff_date: Separate articles older than cutoff_date
    """

    assert p.isfile(input_file) and input_file.endswith('.jl')

    output_dir = p.abspath(output_dir)
    if not p.isdir(output_dir):
        os.mkdir(output_dir)

    old_news = p.join(output_dir, 'old_news')
    if not p.isdir(old_news):
        os.mkdir(old_news)

    t_0 = time()
    new, old = 0, 0
    with open(input_file, 'r') as srcf:
        for line in srcf:
            article = json.loads(line)
            pubdate = article['lexisnexis']['metadata']['published_date']
            if pubdate >= cutoff_date:
                targetf = p.join(output_dir, f'{pubdate}.jl')
                new += 1
            else:
                targetf = p.join(old_news, f'{pubdate}.jl')
                old += 1
            with open(targetf, 'a') as trgf:
                trgf.write(f'{json.dumps(article)}\n')

    print(f'Sorted {new+old} files in {time()-t_0:0.1f}s '
          f'({100*new/(new+old):0.1f}% published since {cutoff_date})')
