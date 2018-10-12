import requests
from optparse import OptionParser

options = OptionParser()
options.add_option('-i', '--index_path')
(opts, _) = options.parse_args()

local_url = 'http://localhost:5954/faiss'
payload = {'path': opts.index_path}
r = requests.get(local_url, params=payload)
print(r.text)
