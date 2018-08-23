import requests
local_url = 'http://localhost:5555/search'
payload = {'query': 'what is the moving annual return'}
r = requests.get(local_url, params=payload)
print(r.text)