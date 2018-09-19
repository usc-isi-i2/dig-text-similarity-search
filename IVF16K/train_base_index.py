import os
import sys
import faiss
import numpy as np
from time import time
from optparse import OptionParser
# <editor-fold desc="Parse Options">
cwd = os.path.abspath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
tmp_emb_dir = os.path.join(cwd, '../data/vectorized_sage_news/new_2018-08-from07to13')
tmp_index_dir = os.path.join(cwd, '../saved_indexes/IVF16K_indexes')

arg_parser = OptionParser()
arg_parser.add_option('-i', '--input_npz_dir', default=tmp_emb_dir)
arg_parser.add_option('-o', '--output_index_dir', default=tmp_index_dir)
arg_parser.add_option('-b', '--base_empty_index', default='emptyTrainedIVF16384.index')
arg_parser.add_option('-n', '--n_training', type='int', default=100)
(args, _) = arg_parser.parse_args()
# </editor-fold>

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from digtextsimilaritysearch.vectorizer.sentence_vectorizer \
    import SentenceVectorizer


"""
Script for training a base IVF index for later use.

Options:
    -i  Full path to .npz directory
    -o  Full path to index directory
    -b  Name of empty index to be trained
    -n  Number of .npz files to train on
"""


t_start = time()

# Set up paths
emb_dir = args.input_npz_dir
assert os.path.isdir(emb_dir), 'Full path does not exist: {}'.format(emb_dir)
index_dir = args.output_index_dir
if not os.path.isdir(index_dir):
    os.mkdir(index_dir)
trained_base_index = os.path.join(index_dir, args.base_empty_index)

n_train = args.n_training
training_path = os.path.join(emb_dir, 'training_set{}.npz'.format(n_train))


# Init
t_init0 = time()
sv = SentenceVectorizer()
t_init1 = time()
print('\nInitialized in {:0.2f}s\n'.format(t_init1-t_init0))


# Make training set (use first 100 chunks by default)
if not os.path.isfile(training_path):
    small_npzs = list()
    for (dir_path, _, file_list) in os.walk(emb_dir):
        for f in file_list:
            if f.endswith('.npz') and not f.startswith('train'):
                small_npzs.append(os.path.join(dir_path, f))
        break
    small_npzs.sort()
    print('{} chunk.npz paths found'.format(len(small_npzs)))

    training_set = list()
    print('Using {} chunks for training'.format(n_train))
    t_gather0 = time()
    for npz in small_npzs[:n_train]:
        training_embs, _ = sv.load_vectors(npz)
        training_set.append(training_embs)
    training_set = np.vstack(training_set)
    np.savez_compressed(training_path, training_set=training_set)
    t_gather1 = time()
    print('Training set stacked and saved in {:0.2f}s'.format(t_gather1-t_gather0))
else:
    loaded_set = np.load(training_path)
    training_set = loaded_set['training_set']
    print('Training set loaded')


# Create index
index_type = 'IVF16384,Flat'
print('\nCreating index: {}'.format(index_type))
index = faiss.index_factory(training_set.shape[1], index_type)

# Train
print(' Training index...')
t_train0 = time()
index.train(training_set)
t_train1 = time()
print(' Index trained in {:0.2f}s'.format(t_train1-t_train0))

# Save
print(' Saving index...')
faiss.write_index(index, trained_base_index)
print(' Index saved...')

t_end = time()
print('\nProcess completed in {:0.2f}s'.format(t_end-t_start))
