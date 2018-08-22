# dig-text-similarity-search


## Virtual Environment
#### Initialize:
```
conda env create .
source activate dig_text_similarity
conda install -c pytorch faiss-cpu
source activate dig_text_similarity
```

#### Deactivate:
```
source deactivate
```


## HBase
#### Run hbase docker (for test purposes only):
`To persist data, use the -v option`
```
docker pull dajobe/hbase
docker run -t -i -p 9001:9001 -p 9090:9090 -p 2181:2181 -v /tmp/hbase_data:/data --rm dajobe/hbase
```

#### Connect to docker hbase using the hbase_adapter code in this repo:
```
hb = HBaseAdapter('localhost')
```


## Instructions to run initial code
#### Collector class in query_index_handler.py
The Collector is primarily a proof of concept. 
```
query_handler = Collector(path_to_index_dir=/full/path/to/project/saved_indexes, 
                          grand_index_name=primary_saved.index, 
                          base_index_name=secondary_saved.index, 
                          path_to_model=/full/path/to/saved/UniversalSentenceEncoder or URL)
```
Collector.add_to_index(docs) demonstrates how TensorFlow -> Faiss Index 
can be connected end-to-end (i.e. sentence strings to searchable index). 
Then use Collector.query_index(query_str) to find similar results.

#### vectorize_en_mass.py
This script uses specific methods from the Collector class. 
It reads cdr_docs with a "split_sentences" field, calculates the 
vector embedding for each sentence, and saves each sent_dict as 
json.dumps() in a large .json, line-by-line.


## Installing TensorFlow from Source (for CPU)
TF Docs: https://www.tensorflow.org/install/install_sources <br />

#### Setup:
```
bazel version
brew upgrade bazel
```

#### Clone TensorFlow:
```
git clone https://github.com/tensorflow/tensorflow 
cd tensorflow
git checkout r1.9
```

#### Configure for CPU:
Ensure dig_text_similarity env is active.
```
./configure
```
Add XLA support. Set everything else to default.
```
Please specify the location of python. [Default is /anaconda3/envs/MyEnv/bin/python]: 
...
Do you wish to build TensorFlow with XLA JIT support? [y/N]: y 
...
```

#### Build:
Note gcc version:
```
gcc -v
```
* If gcc 4: 
    ```
    bazel build --config=mkl --config=opt //tensorflow/tools/pip_package:build_pip_package
    ```
* If gcc 5 or later:
    ```
    bazel build --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" --config=mkl --config=opt //tensorflow/tools/pip_package:build_pip_package
    ```
This will take a long time...

#### Create Package and Install:
```
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
pip install /tmp/tensorflow_pkg/tensorflow-*.whl
```
