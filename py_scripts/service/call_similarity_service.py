import requests
from datetime import date
from argparse import ArgumentParser

arp = ArgumentParser(description='Search the index with any query')
default_query = 'Do Elon Musk\'s tweets help Tesla stock?'
arp.add_argument('-q', '--query', default=default_query,
                 help=f'Cohesive sentences provide better results '
                      f'(Default: {default_query})')
arp.add_argument('-s', '--start_date', default='2019-03-01',
                 help='Earliest ISO formatted publication date to search '
                      '(Default: 2019-03-01)')
arp.add_argument('-e', '--end_date', default=date.today().isoformat(),
                 help=f'Final ISO formatted publication date to search '
                      f'(Default: Today {date.today().isoformat()})')
opts = arp.parse_args()


local_url = 'http://localhost:5954/search'
payload = {'query': opts.query,
           'start_date': opts.start_date,
           'end_date': opts.end_date}
r = requests.get(local_url, params=payload)
print(r.text)
