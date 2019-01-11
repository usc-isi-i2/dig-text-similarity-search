import os.path as p

__all__ = ['std_config', 'lrg_config']


"""
Note: Universal Sentence Encoder has two different embedding spaces
      'large_emb_space' = ...
            False --> Deep Averaging Network 
            True  --> Transformer Network
"""

# Standard config
std_config = {
    'faiss_index_path': p.abspath('../data/shards/'),
    'large_emb_space': False,   # Model & index emb space must match
    'host': '0.0.0.0',
    'port': '5954',
    'es_endpoint': 'http://mydig-sage-internal.isi.edu/es'
}

# Large config
lrg_config = dict(std_config)
lrg_config['large_emb_space'] = True
