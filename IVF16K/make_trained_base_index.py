import os
import sys
import faiss
import numpy as np
from time import time
from digtextsimilaritysearch.vectorizer.sentence_vectorizer \
    import SentenceVectorizer


t_start = time()

# Set up paths
cwd = os.getcwd()
emb_dir = os.path.join(cwd, 'data/vectorized_sage_news/new_2018-08-from07to13')
index_dir = os.path.join(cwd, 'saved_indexes')
training_path = os.path.join(emb_dir, 'training_set.npz')
trained_base_index = os.path.join(index_dir, 'emptyTrainedIVF16384.index')

small_npzs = list()
for (dir_path, _, file_list) in os.walk(emb_dir):
    for f in file_list:
        if f.endswith('.npz'):
            small_npzs.append(os.path.join(dir_path, f))
    break
small_npzs.sort()
print('{} chunk.npz paths found'.format(len(small_npzs)))

# Init
t_init0 = time()
sv = SentenceVectorizer()
t_init1 = time()
print('Initialized in {:0.2f}s'.format(t_init1-t_init0))


# Make training set (use first 100 chunks by default)
training_set = list()
n_train = int(sys.argv[1]) if len(sys.argv) > 1 else 100
print('Using {} chunks for training'.format(n_train))
t_gather0 = time()
for npz in small_npzs[:n_train]:
    training_embs, _ = sv.load_vectors(npz)
    training_set.append(training_embs)
training_set = np.vstack(training_set)
t_gather1 = time()
print('Training set stacked in {:0.2f}s'.format(t_gather1-t_gather0))


# Create index
index_type = 'IVF16384,Flat'
index = faiss.index_factory(training_set.shape[1], index_type)

# Train
t_train0 = time()
index.train(training_set)
t_train1 = time()
print('Index trained in {:0.2f}s'.format(t_train1-t_train0))

# Save
faiss.write_index(index, trained_base_index)

t_end = time()
print('\nProcess completed in {:0.2f}s'.format(t_end-t_start))
