import requests
from argparse import ArgumentParser

arp = ArgumentParser(description='')
arp.add_argument('-q', '--query', default='What is your Quest?',
                 help='Search with your own query.')
(opts, _) = arp.parse_args()


local_url = 'http://localhost:5954/search'
payload = {'query': opts.query}
r = requests.get(local_url, params=payload)
print(r.text)
