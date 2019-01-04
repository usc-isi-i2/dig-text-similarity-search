import requests
from argparse import ArgumentParser

arp = ArgumentParser(description='Deploy a faiss index.')
arp.add_argument('index_path', help='Path to faiss index to be added.')
arp.add_argument('-u', '--url', default='http://localhost:5954/faiss',
                 help='Port handling similarity server.')
(opts, _) = arp.parse_args()


def add_shard(path=None, url='http://localhost:5954/faiss'):
    payload = {'path': path}
    r = requests.put(url, params=payload)
    print(r.text)


if __name__ == '__main__':
    if opts.index_path:
        add_shard(path=opts.index_path, url=opts.url)
    else:
        print('Please provide path to a faiss index.')
