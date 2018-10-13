import requests
from optparse import OptionParser

options = OptionParser()
options.add_option('-i', '--index_path', default=None)
options.add_option('-u', '--url', default='http://localhost:5954/faiss')
(opts, _) = options.parse_args()


def add_shard(path=None, url='http://localhost:5954/faiss'):
    payload = {'path': path}
    r = requests.get(url, params=payload)
    print(r.text)


if __name__ == '__main__':
    if opts.index_path:
        add_shard(path=opts.index_path, url=opts.url)
    else:
        print('Please provide path to new shard with: -i /path/to/shard.index')
