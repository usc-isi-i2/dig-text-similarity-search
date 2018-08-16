# dig-text-similarity-search

## Initialize virtual environment
```
python3 -m venv dig_text
source dig_text/bin/activate
pip install -r requirements.txt
```

## Run hbase docker (for test purposes only)
`To persist data, use the -v option`
```
docker pull dajobe/hbase
docker run -t -i -p 9001:9001 -p 9090:9090 -p 2181:2181 -v /tmp/hbase_data:/data --rm dajobe/hbase
```
### Connect to docker hbase using the hbase_adapter code in this repo
```
hb = HBaseAdapter('localhost')
```