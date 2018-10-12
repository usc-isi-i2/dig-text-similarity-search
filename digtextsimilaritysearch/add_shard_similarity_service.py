import requests
from optparse import OptionParser

options = OptionParser()
options.add_option('-i', '--index_path', default=None)
(opts, _) = options.parse_args()


def add_shard(path=opts.index_path, local_url='http://localhost:5954/faiss'):
    payload = {'path': path}
    r = requests.get(local_url, params=payload)
    print(r.text)


if __name__ == '__main__':
    if opts.index_path:
        add_shard(path=opts.index_path)
    else:
        print('Please provide path to new shard with: -i /path/to/shard.index')
