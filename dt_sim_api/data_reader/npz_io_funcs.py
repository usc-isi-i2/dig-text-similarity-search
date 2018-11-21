import os
from time import time
from typing import List, Tuple, Union

import numpy as np

__all__ = ['get_all_npz_paths',
           'load_training_npz',
           'load_with_ids', 'save_with_ids']


##### Misc #####
def get_all_npz_paths(npz_parent_dir: str) -> List[str]:
    """
    Finds all .npz files nested anywhere within a parent directory.

    :param npz_parent_dir: Parent directory of .npz files
    :return: List of full paths to .npz files sorted alphabetically
    """
    assert os.path.isdir(npz_parent_dir), \
        'Input Error: {} must point to an existing directory' \
        ''.format(npz_parent_dir)

    npz_paths = list()
    for (dirpath, _, filenames) in os.walk(npz_parent_dir, topdown=True):
        for f in filenames:
            if f.endswith('.npz'):
                npz_paths.append(os.path.abspath(os.path.join(dirpath, f)))
    return sorted(npz_paths)


##### Load .npz #####
def load_training_npz(npz_paths: List[str], training_set_name: str,
                      mmap_tmp: bool = True) -> np.array:
    """
    Merges .npz files into a memory mapped, numpy array for training a
    base faiss index.

    :param npz_paths: List of full paths to .npz files
    :param training_set_name: Filename for training_set
    :param mmap_tmp: Bool to load component .npz files in mmap mode
    :return: Memory mapped training set array
    """
    t_load = time()
    emb_list = list()
    emb_lens = list()
    for npzp in npz_paths:
        emb, _, _ = load_with_ids(npzp, mmap=mmap_tmp)
        emb_list.append(emb), emb_lens.append(emb.shape)

    tot_embs = sum([n[0] for n in emb_lens])
    emb_wide = emb_lens[0][1]
    print('\nFound {} vectors of {}d'.format(tot_embs, emb_wide))

    training_set_name = os.path.abspath(training_set_name)
    ts_memmap = np.memmap(training_set_name, dtype=np.float32,
                          mode='w+', shape=(tot_embs, emb_wide))

    place = 0
    for emb in emb_list:
        n_vect = emb.shape[0]
        ts_memmap[place:place+n_vect, :] = emb[:]
        place += n_vect

    m, s = divmod(time()-t_load, 60)
    print(' Training set loaded in {}m{:0.2f}s'.format(int(m), s))

    return ts_memmap


def load_with_ids(file_path: str, mmap: bool = True, load_sents=False
                  ) -> Union[Tuple[np.array, np.array],
                             Tuple[np.array, np.array, np.array]]:
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
        return embeddings, sent_ids


##### Save .npz #####
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
        'Error: len mismatch. Found {} embeddings and {} sent_ids' \
        ''.format(len(embeddings), len(sent_ids))

    # Save
    if compressed:
        np.savez_compressed(file=file_path, embeddings=embeddings,
                            sent_ids=sent_ids, sentences=sentences)
    else:
        np.savez(file=file_path, embeddings=embeddings,
                 sent_ids=sent_ids, sentences=sentences)
