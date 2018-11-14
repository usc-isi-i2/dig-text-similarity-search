import os
import json
from time import time
from typing import List, Tuple, Union

import numpy as np


##### Read news.jl #####

def check_all_docs(file_path: str, b_size: int = 512*128
                   ) -> (int, int, int, int):
    doc_count = 0
    line_count = 0
    junk_count = 0
    with open(file_path, 'r') as jl:
        for doc in jl:
            document = json.loads(doc)
            content = document['lexisnexis']['doc_description']
            if content and not content == '' and not content == 'DELETED_STORY' \
                    and 'split_sentences' in document and len(document['split_sentences']):
                doc_count += 1
                line_count += len(document['split_sentences']) + 1
            else:
                junk_count += 1
    n_batches = divmod(line_count, b_size)[0] + 1
    return doc_count, line_count, junk_count, n_batches


def check_training_docs(file_path: str, b_size: int = 512*128
                        ) -> (int, int, int, int, int, int):
    doc_count = 0
    line_count = 0
    good_sents = 0
    junk_count = 0
    with open(file_path, 'r') as jl:
        for doc in jl:
            document = json.loads(doc)
            content = document['lexisnexis']['doc_description']
            if content and not content == '' and not content == 'DELETED_STORY' \
                    and 'split_sentences' in document and len(document['split_sentences']):
                doc_count += 1
                line_count += len(document['split_sentences']) + 1
                for sent in document['split_sentences']:
                    if len(sent) > 20:
                        if len(sent.split(' ')) > 3:
                            good_sents += 1
            else:
                junk_count += 1
    n_batches = int(divmod(line_count, b_size)[0]) + 1
    n_good_batches = int(divmod(good_sents, b_size)[0]) + 1
    return doc_count, line_count, good_sents, junk_count, n_batches, n_good_batches


##### Load news.jl #####

def aggregate_all_docs(file_path: str, b_size: int = 512*128
                       ) -> (List[str], np.array):
    batched_text = list()
    batched_ids = list()
    with open(file_path, 'r') as jl:
        for doc in jl:
            document = json.loads(doc)
            content = document['lexisnexis']['doc_description']
            if content and not content == '' and not content == 'DELETED_STORY' \
                    and 'split_sentences' in document and len(document['split_sentences']):
                text = list()
                text.append(document['lexisnexis']['doc_title'])
                text.extend(document['split_sentences'])

                doc_id = document['doc_id']
                base_sent_id = np.int64(doc_id + '0000')
                sent_ids = list()
                for jj, _ in enumerate(text):
                    sent_ids.append(base_sent_id + jj)
                sent_ids = np.vstack(sent_ids).astype(np.int64)
                assert sent_ids.shape[0] == len(text), \
                    'Something went wrong while making sent_ids'

                batched_text.extend(text)
                batched_ids.append(sent_ids)

            if len(batched_text) >= b_size:
                batched_ids = np.vstack(batched_ids).astype(np.int64)
                yield batched_text, batched_ids
                batched_text = list()
                batched_ids = list()

    batched_ids = np.vstack(batched_ids).astype(np.int64)
    yield batched_text, batched_ids


def aggregate_training_docs(file_path: str, b_size: int = 512*128
                            ) -> (List[str], np.array):
    batched_text = list()
    batched_ids = list()
    with open(file_path, 'r') as jl:
        for doc in jl:
            document = json.loads(doc)
            content = document['lexisnexis']['doc_description']
            if content and not content == '' and not content == 'DELETED_STORY' \
                    and 'split_sentences' in document and len(document['split_sentences']):
                text = list()
                all_text = list()

                if len(document['lexisnexis']['doc_title']) > 5:
                    text.append(document['lexisnexis']['doc_title'])
                all_text.append(document['lexisnexis']['doc_title'])
                for sent in document['split_sentences']:
                    if len(sent) > 20:
                        if len(sent.split(' ')) > 3:
                            text.append(sent)
                    all_text.append(sent)

                if len(text):
                    sent_ids = list()
                    doc_id = document['doc_id']
                    base_sent_id = np.int64(doc_id + '0000')
                    for jj, a_sent in enumerate(all_text):
                        if a_sent in text:
                            sent_ids.append(base_sent_id + jj)
                    sent_ids = np.vstack(sent_ids).astype(np.int64)

                    if not sent_ids.shape[0] == len(text):
                        print(sent_ids.shape)
                        print(len(text))

                        if sent_ids.shape[0] > len(text):
                            print('Truncating ids')
                            sent_ids = sent_ids[:len(text)]
                        else:
                            print('Making fake ids')
                            sent_ids = list()
                            for jjj, _ in enumerate(text):
                                sent_ids.append(base_sent_id + jjj)
                            sent_ids = np.vstack(sent_ids).astype(np.int64)

                    assert sent_ids.shape[0] == len(text), \
                        'Something went wrong while making sent_ids'

                    batched_text.extend(text)
                    batched_ids.append(sent_ids)

            if len(batched_text) >= b_size:
                batched_ids = np.vstack(batched_ids).astype(np.int64)
                yield batched_text, batched_ids
                batched_text = list()
                batched_ids = list()

    batched_ids = np.vstack(batched_ids).astype(np.int64)
    yield batched_text, batched_ids


##### Save data.npz #####

# TODO: Depreciate save_vectors
def save_vectors(embeddings: List[tf.Tensor], sentences: List[object],
                 file_path: str):
    """
    Converts embedding tensors and corresponding list of sentences into np.arrays,
    then saves both arrays in the same compressed .npz file

    Note: .npz will be appended to file_path automatically

    :param embeddings: List of tensors made from sentences
    :param sentences: List of corresponding sentences
    :param file_path: /full/path/to/file_name
    :return: Writes zipped archive to disk
    """
    embeddings = np.array(embeddings, dtype=np.float32)
    sentences = np.array(sentences, dtype=object)
    np.savez_compressed(file=file_path, embeddings=embeddings, sentences=sentences)


def save_with_ids(file_path: str, embeddings, sentences, sent_ids,
                  compressed=True):
    # Format
    if not isinstance(embeddings, np.ndarray):
        embeddings = np.vstack(embeddings).astype(np.float32)
    if not isinstance(sentences, np.ndarray):
        sentences = np.array(sentences, dtype=np.str)
    if not isinstance(sent_ids, np.ndarray):
        try:
            sent_ids = np.array(sent_ids, dtype=np.int64)
        except ValueError:
            print(sent_ids)

    # Save
    if compressed:
        np.savez_compressed(file=file_path,
                            sent_ids=sent_ids, embeddings=embeddings, sentences=sentences)
    else:
        np.savez(file=file_path,
                 sent_ids=sent_ids, embeddings=embeddings, sentences=sentences)


##### Load data.npz #####

# TODO: Depreciate load_vectors
def load_vectors(file_path: str, mmap: bool = True
                 ) -> (np.array, np.array):
    """
    Loads zipped archive containing embeddings and sentences as two separate np.arrays

    :param file_path: /full/path/to/file_name.npz
    :param mmap: If mmap is True, the .npz file is not read from disk, not into memory
    :return: embeddings and sentences as separate np.arrays
    """
    if not file_path.endswith('.npz'):
        file_path += '.npz'
    if mmap:
        mode = 'r'
    else:
        mode = None

    loaded = np.load(file=file_path, mmap_mode=mode)
    embeddings = loaded['embeddings']
    sentences = loaded['sentences']
    return embeddings, sentences


def load_with_ids(file_path: str, mmap: bool = True
                  ) -> (np.array, np.array, np.array):
    if not file_path.endswith('.npz'):
        file_path += '.npz'
    if mmap:
        mode = 'r'
    else:
        mode = None

    loaded = np.load(file=file_path, mmap_mode=mode)
    embeddings = loaded['embeddings']
    sentences = loaded['sentences']
    sent_ids = loaded['sent_ids']
    return embeddings, sentences, sent_ids


def load_training_npz(npz_paths: List[str], tmp_name: str,
                      mmap: bool = True) -> np.array:
    t_load = time()

    emb_list = list()
    emb_lens = list()
    for npzp in npz_paths:
        emb, _, _ = load_with_ids(npzp, mmap=mmap)
        emb_list.append(emb), emb_lens.append(emb.shape)

    tot_embs = sum([n[0] for n in emb_lens])
    emb_wide = emb_lens[0][1]
    print('\nFound {} vectors of {}d'.format(tot_embs, emb_wide))

    ts_memmap = np.memmap(tmp_name, dtype=np.float32,
                          mode='w+', shape=(tot_embs, emb_wide))

    place = 0
    for emb in emb_list:
        n_vect = emb.shape[0]
        ts_memmap[place:place+n_vect, :] = emb[:]
        place += n_vect

    m, s = divmod(time()-t_load, 60)
    print(' Training set loaded in {}m{:0.2f}s'.format(int(m), s))

    return ts_memmap


##### Misc #####

def get_all_npz_paths(npz_parent_dir: str) -> List[str]:
    npz_paths = list()
    for (dirpath, _, filenames) in os.walk(npz_parent_dir, topdown=True):
        for f in filenames:
            if f.endswith('.npz'):
                npz_paths.append(os.path.join(dirpath, f))
    return sorted(npz_paths)


def check_unique(path: str, i=0) -> str:
    if os.path.exists(path):
        print('\nWarning: File already exists  {}'.format(path))
        path = path.split('.')
        path = path[0] + '_{}.'.format(i) + path[-1]
        print('         Testing new path  {}\n'.format(path))
        i += 1
        check_unique(path=path, i=i)
    return path


def clear(tmp_dir_path: str):
    for (tmp_dir, _, tmp_files) in os.walk(tmp_dir_path):
        for file in tmp_files:
            os.remove(os.path.join(tmp_dir, file))
    os.rmdir(tmp_dir_path)
