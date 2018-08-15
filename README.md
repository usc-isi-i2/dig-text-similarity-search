# dig-text-similarity-search

## Initialize virtual environment
```
python3 -m venv dig_text
source dig_text/bin/activate
pip install -r requirements.txt
```

## Run hbase docker (for test purposes only)
```
docker pull iwan0/hbase-thrift-standalone
docker run -t -i -p 9001:9001 -p 9090:9090 -p 2181:2181 --rm iwan0/hbase-thrift-standalone
```
### Connect to docker hbase using the hbase_adapter code in this repo
```
hb = HBaseAdapter('localhost')
```