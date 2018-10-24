# dig-text-similarity-search

## Overview
#### Text Search without Keywords:
This is a search engine for ranking news articles from LexisNexis 
using sentence vectors rather than key words. 


#### Basic Recipe:
1) Prepare text corpus as sentences with int ids
2) Vectorize sentences with Google's [Universal Sentence Encoder](https://tfhub.dev/google/universal-sentence-encoder/2)
3) Put vectors into a searchable [Faiss index](https://github.com/facebookresearch/faiss)
4) Search with vectorized query


## Virtual Environment
#### Initialize:
```
conda env create .
source activate dig_text_similarity
ipython kernel install --user --name=dig_text_similarity
```

#### Deactivate:
```
source deactivate
```


## Usage
Vectorized similarity search does not use key-words. To query the corpus properly, please provide complete 
sentences as input. 

#### To get started:
Create an on-disk searchable faiss index by running [`streaming_preprocessor.py`](https://github.com/usc-isi-i2/dig-text-similarity-search/blob/master/preprocessing/streaming_preprocessor.py):
```
source activate dig_text_similarity
python preprocessing/streaming_preprocessor.py -i data/example/ -o saved_indexes/ -r -d
```

Note: Every faiss shard should contain absolute partitions of the sentences within the corpus. Using 
multiple shards that share duplicate `faiss_ids` may give unexpected results. 

#### Query vectorization with docker:
Before running [`similarity_server.py`](https://github.com/usc-isi-i2/dig-text-similarity-search/blob/master/digtextsimilaritysearch/similarity_server.py), 
download docker and run two shell scripts:

1) `$ ./digtextsimilaritysearch/vectorizer/prep_service_model.sh` ([link to script](https://github.com/usc-isi-i2/dig-text-similarity-search/blob/master/digtextsimilaritysearch/vectorizer/prep_service_model.sh))
2) `$ ./digtextsimilaritysearch/vectorizer/run_service_model.sh` ([link to script](https://github.com/usc-isi-i2/dig-text-similarity-search/blob/master/digtextsimilaritysearch/vectorizer/run_service_model.sh))

The first script will encapsulate the Universal Sentence Encoder (Deep Averaging Network v2) in a suitable 
form for running in a docker container, and the second script runs the container locally through port `8501` 
for query vectorization.

Note: Although it is possible to do so, it is not recommended to use the dockerized model for vectorizing
batches of sentences during preprocessing.

#### Configuration:
In [`digtextsimilaritysearch/`](https://github.com/usc-isi-i2/dig-text-similarity-search/tree/master/digtextsimilaritysearch), 
the file [`config.py`](https://github.com/usc-isi-i2/dig-text-similarity-search/blob/master/digtextsimilaritysearch/config.py) 
holds configuration instructions for running [`similarity_server.py`](https://github.com/usc-isi-i2/dig-text-similarity-search/blob/master/digtextsimilaritysearch/similarity_server.py). 

Note: `config["faiss_index_path"]` should be an absolute path to the directory containing your 
`{shard_name}.index` files (the DeployShards [index handler](https://github.com/usc-isi-i2/dig-text-similarity-search/blob/master/digtextsimilaritysearch/indexer/IVF_disk_index_handler.py) 
will load every shard in the directory). Change this path if your faiss index shards are saved elsewhere.

#### Similarity service:
Run [`similarity_server.py`](https://github.com/usc-isi-i2/dig-text-similarity-search/blob/master/digtextsimilaritysearch/similarity_server.py) 
and test it with [`call_similarity_service.py`](https://github.com/usc-isi-i2/dig-text-similarity-search/blob/master/digtextsimilaritysearch/call_similarity_service.py). 
Use the command line argument `-q "What was the question again?"` to input a different query.

Ex: 
``` 
$ python digtextsimilaritysearch/call_similarity_service.py -q "What is the air-speed velocity of an unladen swallow?"
```

Note: The DocumentProcessor is constructed with `storage_adapter=None`, so `dp.query_text()` will return 
`faiss_ids` and their approximate difference scores (L2) relative to the query vector. 

#### Storage (depreciated): 
Since faiss can only store int64 ids, the storage adapter links these ids to the actual text of the 
results. Currently, this step of linking `faiss_ids` to sentences within documents is handled outside of 
the document processor. 

Note: All storage components in this repo will be depreciated soon. 
