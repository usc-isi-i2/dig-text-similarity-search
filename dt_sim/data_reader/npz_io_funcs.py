import os.path as p
from time import time
from pathlib import Path
from typing import Tuple, Union

import numpy as np

__all__ = ['load_training_npz',
           'load_with_ids', 'save_with_ids']


Embeddings_IDs_Sents = Tuple[np.array, np.array, np.array]


#### Load .npz ####
def load_training_npz(training_set_path: str, npz_dir: str = None,
                      n_vectors: int = 1000000, dim: int = 512) -> np.array:
    """
    Merges .npz files into a memory mapped, numpy array for training a
    base faiss index.

    :param training_set_path: Path to training_set.dat
    :param npz_dir: Parent dir containing .npz files
    :param n_vectors: Number of vectors to put into training set
    :param dim: Embedding dimensionality
    :return: Memory mapped training set array
    """
    t_load = time()
    training_set_path = p.abspath(training_set_path)
    if p.exists(training_set_path):
        # Load
        ts_memmap = np.memmap(training_set_path, dtype=np.float32,
                              mode='r', shape=(n_vectors, dim))
    elif npz_dir:
        # Find
        npz_paths = list(Path(npz_dir).glob('*.npz'))
        print(f'Found {len(npz_paths)} .npz files')

        # Empty
        ts_memmap = np.memmap(training_set_path, dtype=np.float32,
                              mode='w+', shape=(n_vectors, dim))

        # Populate
        npz_count = 0
        emb_count = 0
        while emb_count < n_vectors:
            emb, _, _ = load_with_ids(npz_paths[npz_count])
            emb_batch = emb_count + emb.shape[0]
            npz_count += 1
            print(f'Loaded {emb_batch}/{n_vectors} vectors '
                  f'of {emb.shape[1]}d from {npz_count} files...')
            if emb_batch > n_vectors:
                stop = n_vectors - emb_count
                ts_memmap[emb_count:n_vectors, :] = emb[:stop, :]
            else:
                ts_memmap[emb_count:emb_batch, :] = emb[:, :]
            emb_count += emb.shape[0]

        # Write
        ts_memmap.flush()
    else:
        print('Nothing to load')
        return

    # Format
    training_set = np.ndarray(buffer=ts_memmap[:n_vectors],
                              dtype=np.float32,
                              shape=(n_vectors, dim))

    m, s = divmod(time()-t_load, 60)
    print(f'Training set loaded in {int(m)}m{s:0.2f}s')
    return training_set


def load_with_ids(file_path: Union[str, Path], mmap: bool = True,
                  load_sents=False) -> Embeddings_IDs_Sents:
    """
    Load preprocessed sentence embeddings with corresponding integer ids.
    Note: Loading sentences is optional

    :param file_path: File to load (.npz)
    :param mmap: Flag to load file as a memory mapped array
    :param load_sents: Flag to return saved sentences (may be '')
    :return: Numpy arrays containing embeddings & ids (and possibly sentences)
    """
    if not file_path.endswith('.npz'):
        file_path += '.npz'
    if mmap:
        mode = 'r'
    else:
        mode = None

    loaded = np.load(file=file_path, mmap_mode=mode)
    embeddings = loaded['embeddings']
    sent_ids = loaded['sent_ids']
    sentences = loaded['sentences']
    if load_sents:
        return embeddings, sent_ids, sentences
    else:
        return embeddings, sent_ids, np.array([['']], dtype=np.str)


#### Save .npz ####
def save_with_ids(file_path: str, embeddings, sent_ids,
                  sentences='', compressed: bool = True):
    """
    Save preprocessed sentence embeddings with corresponding integer ids.
    Note: Saving sentences is optional

    :param file_path: File to save (numpy can handle file extension)
    :param embeddings: Sentence vectors to save
    :param sent_ids: Corresponding faiss ids
    :param sentences: Corresponding sentences (optional)
    :param compressed: Flag to compress output files (takes longer)
    """
    # Format
    if not isinstance(embeddings, np.ndarray):
        embeddings = np.vstack(embeddings).astype(np.float32)
    if not isinstance(sent_ids, np.ndarray):
        try:
            sent_ids = np.array(sent_ids, dtype=np.int64)
        except ValueError as value_error:
            print(sent_ids)
            raise value_error
    if not isinstance(sentences, np.ndarray):
        sentences = np.array(sentences, dtype=np.str)

    assert len(embeddings) == len(sent_ids), \
        f'Error: len mismatch. ' \
        f'Found {len(embeddings)} embeddings and {len(sent_ids)} sent_ids'

    # Save
    if compressed:
        np.savez_compressed(file=file_path, embeddings=embeddings,
                            sent_ids=sent_ids, sentences=sentences)
    else:
        np.savez(file=file_path, embeddings=embeddings,
                 sent_ids=sent_ids, sentences=sentences)
