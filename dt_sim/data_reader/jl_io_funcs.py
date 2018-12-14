import json
from typing import List, Tuple

import numpy as np

__all__ = ['check_all_docs', 'get_all_docs',
           'check_training_docs', 'get_training_docs']


"""
    Funcs for reading LexisNexis news dumps as JSON Lines files.
    
    Assumes JSON Structure: 
        - News story title: doc['lexisnexis']['doc_title']
        - News story text: doc['lexisnexis']['doc_description']
        - News story text (split into sentences): doc['split_sentences']
        - Unique integer doc id (created internally for faiss): doc['doc_id']
"""


##### Read news.jl #####
def check_all_docs(jl_file_path: str, batch_size: int = 0
                   ) -> Tuple[int, int, int, int]:
    """
    Reads input news.jl file and returns counts of entities found.

    :param jl_file_path: Path to LexisNexis news.jl
    :param batch_size: Number of sentences per batch
    :return:
        doc_count: Total number of good documents in file
        line_count: Total number of sentences (+ 1 for title) in all docs
        junk_count: Total number of bad documents in file
        n_batches: Total number of batches file will yield given batch_size > 0
    """
    doc_count = 0
    line_count = 0
    junk_count = 0
    with open(jl_file_path, 'r') as jl:
        for doc in jl:
            document = json.loads(doc)
            content = document['lexisnexis']['doc_description']
            if content and not content == '' and not content == 'DELETED_STORY' \
                    and 'split_sentences' in document and len(document['split_sentences']):
                doc_count += 1
                line_count += len(document['split_sentences']) + 1
            else:
                junk_count += 1
    if batch_size:
        n_batches = int(divmod(line_count, batch_size)[0]) + 1
    else:
        n_batches = 0
    return doc_count, line_count, junk_count, n_batches


def check_training_docs(jl_file_path: str, batch_size: int = 0,
                        title_char_min: int = 5, sent_len_min: int = 3
                        ) -> Tuple[int, int, int, int, int, int]:
    """
    Reads input news.jl file and returns counts of entities found
    that meet criteria for training a base faiss index.

    :param jl_file_path: Path to LexisNexis news.jl
    :param batch_size: Number of sentences per batch
    :param title_char_min: Ignore titles with fewer characters
    :param sent_len_min: Ignore sentences with fewer words
    :return:
        doc_count: Total number of good documents in file
        line_count: Total number of sentences (+ 1 for title) in all docs
        good_sents: Total number of sentences with more words than sent_len_min
        junk_count: Total number of bad documents in file
        n_batches: Total number of possible batches file could yield given batch_size > 0
        n_good_batches: Total number of good batches file will yield given batch_size > 0
    """
    doc_count = 0
    line_count = 0
    good_sents = 0
    junk_count = 0
    with open(jl_file_path, 'r') as jl:
        for doc in jl:
            document = json.loads(doc)
            content = document['lexisnexis']['doc_description']
            if content and not content == '' and not content == 'DELETED_STORY' \
                    and 'split_sentences' in document and len(document['split_sentences']):
                doc_count += 1
                line_count += len(document['split_sentences']) + 1
                if len(document['lexisnexis']['doc_title']) > title_char_min:
                    good_sents += 1
                for sent in document['split_sentences']:
                    if len(sent) > 20:
                        if len(sent.split(' ')) > sent_len_min:
                            good_sents += 1
            else:
                junk_count += 1
    if batch_size:
        n_batches = int(divmod(line_count, batch_size)[0]) + 1
        n_good_batches = int(divmod(good_sents, batch_size)[0]) + 1
    else:
        n_batches, n_good_batches = 0, 0
    return doc_count, line_count, good_sents, junk_count, n_batches, n_good_batches


##### Load news.jl #####
def get_all_docs(jl_file_path: str, batch_size: int) -> Tuple[List[str], np.array]:
    """
    Reads input news.jl file and yields batches of sentences with matching faiss ids.

    :param jl_file_path: Path to LexisNexis news.jl
    :param batch_size: Number of sentences per batch
    :return: Iterator yielding (batched_text, batched_ids)
    """
    batched_text = list()
    batched_ids = list()
    with open(jl_file_path, 'r') as jl:
        for doc in jl:
            document = json.loads(doc)
            content = document['lexisnexis']['doc_description']
            if isinstance(content, dict):
                content = json.dumps(content)

            if content and not content == '' and not content == 'DELETED_STORY' \
                    and 'split_sentences' in document and len(document['split_sentences']):
                text = list()
                text.append(document['lexisnexis']['doc_title'])
                text.extend(document['split_sentences'])

                # Faiss ids
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

            # Stack and yield batch
            if len(batched_text) >= batch_size:
                batched_ids = np.vstack(batched_ids).astype(np.int64)
                yield batched_text[:batch_size], batched_ids[:batch_size]
                batched_text = list(batched_text[batch_size:])
                batched_ids = list(batched_ids[batch_size:])

    # Yield final batch
    batched_ids = np.vstack(batched_ids).astype(np.int64)
    yield batched_text, batched_ids


def get_training_docs(jl_file_path: str, batch_size: int,
                      title_char_min: int = 5, sent_len_min: int = 3
                      ) -> Tuple[List[str], np.array]:
    """
    Reads input news.jl file and yields batches of sentences with matching faiss ids
    that meet criteria for training a base faiss index.

    :param jl_file_path: Path to LexisNexis news.jl
    :param batch_size: Number of sentences per batch
    :param title_char_min: Ignore titles with fewer characters
    :param sent_len_min: Ignore sentences with fewer words
    :return: Iterator yielding (batched_text, batched_ids)
    """
    batched_text = list()
    batched_ids = list()
    with open(jl_file_path, 'r') as jl:
        for doc in jl:
            document = json.loads(doc)
            content = document['lexisnexis']['doc_description']
            if content and not content == '' and not content == 'DELETED_STORY' \
                    and 'split_sentences' in document and len(document['split_sentences']):
                text = list()
                all_text = list()

                # Skip very short titles
                if len(document['lexisnexis']['doc_title']) > title_char_min:
                    text.append(document['lexisnexis']['doc_title'])
                all_text.append(document['lexisnexis']['doc_title'])

                # Skip short sentences
                for sent in document['split_sentences']:
                    if len(sent) > 20:
                        if len(sent.split(' ')) > sent_len_min:
                            text.append(sent)
                    all_text.append(sent)

                # Faiss ids
                if len(text):
                    sent_ids = list()
                    doc_id = document['doc_id']
                    base_sent_id = np.int64(doc_id + '0000')
                    for jj, a_sent in enumerate(all_text):
                        if a_sent in text:
                            sent_ids.append(base_sent_id + jj)
                    sent_ids = np.vstack(sent_ids).astype(np.int64)

                    # Note: Faiss ids may not be reliable
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

            # Stack and yield batch
            if len(batched_text) >= batch_size:
                batched_ids = np.vstack(batched_ids).astype(np.int64)
                yield batched_text[:batch_size], batched_ids[:batch_size]
                batched_text = list(batched_text[batch_size:])
                batched_ids = list(batched_ids[batch_size:])

    # Yield final batch
    batched_ids = np.vstack(batched_ids).astype(np.int64)
    yield batched_text, batched_ids
