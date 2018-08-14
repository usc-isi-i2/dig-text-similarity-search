import os

from time import time
from doc_loader import load_docs
from query_index_handler import Collector


# TODO: Change paths as necessary
cwd = os.getcwd()
file1 = os.path.join(cwd, "data/testing/1-first-10K-split.jl")
file2 = os.path.join(cwd, "data/testing/2-first-10K-split.jl")
file3 = os.path.join(cwd, "data/testing/3-first-10K-split.jl")

docs = load_docs(file1)
docs.extend(load_docs(file2))
docs.extend(load_docs(file3))

index_dir = os.path.join(cwd, "saved_indexes")
base_idx = "IVF4096Flat.index"
grand_idx = "PCAR256IVF4096SQ8.index"
grand_puba_idx = "OPQ64_192IVF4096SQ8.index"

# TODO: Run following command to specify permanent model location
# $ export TFHUB_CACHE_DIR=/path/to/save/model/in/dir
tf_model = ".../choose/your/path/wisely"
# Note: There is a larger version of the Universal Sentence Encoder
# at "https://www.tensorflow.org/hub/modules/google/universal-sentence-encoder-large/3"
# (with a different embedding space, so vectors/indices need to be recalculated)


collector = Collector(path_to_index_dir=index_dir,
                      base_index_name=base_idx,
                      grand_index_name=grand_idx,
                      path_to_model=tf_model)


def test(doc_list):
    t0 = time()
    collector.add_to_index(doc_list)
    print(time()-t0)


__all__ = ['time', 'load_docs', 'Collector', 'docs', 'collector', 'test']
