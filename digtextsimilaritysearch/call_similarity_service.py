import requests
from optparse import OptionParser


options = OptionParser()
options.add_option('-q', '--query', default='When will the Parker Solar Probe reach perihelion #1?')
(opts, _) = options.parse_args()

local_url = 'http://localhost:5954/search'
payload = {'query': opts.query}
r = requests.get(local_url, params=payload)
print(r.text)
