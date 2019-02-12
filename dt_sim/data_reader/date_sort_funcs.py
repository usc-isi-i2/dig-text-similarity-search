import os
import os.path as p
import json
from time import time
from datetime import date

__all__ = ['pub_date_split']


def pub_date_split(input_file: str, output_dir: str,
                   cutoff_date: str = '0000-00-00',
                   ingest_date: str = str(date.today())):
    """
    Sorts LexisNexis articles by publication date.
    :param input_file: /path/to/LexisNexisCrawlerDump.jl
    :param output_dir: Output destination of pub_date.jl files
    :param cutoff_date: Separate articles older than cutoff_date
    :param ingest_date: Separate articles with erroneous future publication dates
    """

    assert p.isfile(input_file) and input_file.endswith('.jl')

    output_dir = p.abspath(output_dir)
    if not p.isdir(output_dir):
        os.mkdir(output_dir)

    old_news_dir = p.join(output_dir, 'old_news')
    if not p.isdir(old_news_dir):
        os.mkdir(old_news_dir)

    date_error_dir = p.join(output_dir, 'date_error')
    if not p.isdir(date_error_dir):
        os.mkdir(date_error_dir)

    t_0 = time()
    new, old, err, dateless = 0, 0, 0, 0
    with open(input_file, 'r') as srcf:
        for line in srcf:
            article = json.loads(line)

            try:
                event_date = article['knowledge_graph']['event_date'][0]['value'].split('T')[0]
            except KeyError:
                event_date = None

            if event_date and ingest_date >= event_date >= cutoff_date:
                targetf = p.join(output_dir, f'{event_date}.jl')
                new += 1
            elif event_date and event_date < cutoff_date:
                targetf = p.join(old_news_dir, f'{event_date}.jl')
                old += 1
            elif event_date and ingest_date < event_date:
                targetf = p.join(date_error_dir, f'{event_date}.jl')
                err += 1
            else:
                targetf = p.join(date_error_dir, 'dateless_articles.jl')
                dateless += 1

            with open(targetf, 'a') as trgf:
                trgf.write(f'{json.dumps(article)}\n')

    m, s = divmod(time()-t_0, 60)
    print(f'Sorted {new+old+err+dateless} files in {int(m):2d}m{s:0.1f}s '
          f'({100*new/(new+old+err+dateless):0.1f}% published since {cutoff_date})')
