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


#### To get started:
Create an on-disk searchable faiss index [with](https://github.com/usc-isi-i2/dig-text-similarity-search/blob/master/preprocessing/streaming_preprocessor.py):
```
source activate dig_text_similarity
python preprocessing/streaming_preprocessor.py -i data/example/ -o saved_indexes/ -r -d
```


## Virtual Environment
#### Initialize:
```
conda env create .
source activate dig_text_similarity
ipython kernel install --user --name=dig_text_similarity
```
```
Note:
While using this as a pypi package, you may need too perform the following additonal steps:
1) conda install faiss-cpu -c pytorch
2) conda install tensorflow

Also Python venv will not work as a substitute for conda env as there are some conda specific dependencies like faiss-cpu.
So please create virtual environments only using the above method.
```

#### Deactivate:
```
source deactivate
```


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
Please specify the location of python. [Default is /anaconda3/envs/dig_text_similarity/bin/python]: 
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
