import os

__all__ = ['std_config', 'lrg_config']


"""
    Note: Universal Sentence Encoder embedding spaces are non-compatible
          'large_emb_space' = ...
            False --> Deep Averaging Network 
            True --> Transformer Network
"""

#### TEMPLATES ####
# Base config
base_config = {
    'faiss_index_path': '',     # Must specify
    'large_emb_space': False,   # Must match index emb space
    'host': '0.0.0.0',
    'port': '5954',
    'es_endpoint': 'http://mydig-sage-internal.isi.edu/es'
}

# Standard faiss index directory (provide full path)
std_index_path = os.path.abspath('../saved_indexes/shards/')


#### IMPORTABLE CONFIGS ####
# Standard config
std_config = base_config
std_config['faiss_index_path'] = std_index_path
std_config['large_emb_space'] = False

# Large config
lrg_config = base_config
lrg_config['faiss_index_path'] = std_index_path
lrg_config['large_emb_space'] = True
