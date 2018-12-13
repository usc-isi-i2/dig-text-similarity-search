import os.path as p

__all__ = ['std_config', 'lrg_config']


"""
Note: Universal Sentence Encoder has two different embedding spaces
      'large_emb_space' = ...
            False --> Deep Averaging Network 
            True  --> Transformer Network
"""

# Base config
base_config = {
    'faiss_index_path': '',     # Must specify
    'large_emb_space': False,   # Must match index emb space
    'host': '0.0.0.0',
    'port': '5954',
    'es_endpoint': 'http://mydig-sage-internal.isi.edu/es'
}

# Standard faiss index directory (provide full path)
std_index_path = p.abspath('../saved_indexes/shards/')

# Standard config
std_config = dict(base_config)
std_config['faiss_index_path'] = std_index_path

# Large config
lrg_config = dict(base_config)
lrg_config['faiss_index_path'] = std_index_path
lrg_config['large_emb_space'] = True
