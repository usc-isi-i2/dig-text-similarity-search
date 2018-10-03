import os
import sys
import json
import numpy as np
from time import time
from optparse import OptionParser

# <editor-fold desc="Parse Options">
cwd = os.path.abspath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
tmp_split_news = os.path.join(cwd, '../data/testing/split_new_2018-08-08_with_ids.jl')
tmp_output_dir = os.path.join(cwd, '../data/testing/')

options = OptionParser()
options.add_option('-i', '--input_file', default=tmp_split_news)
options.add_option('-o', '--output_dir', default=tmp_output_dir)
options.add_option('-s', '--seed', default='vectorized_')
options.add_option('-n', '--output_filename', default=None)
options.add_option('-b', '--batch_size', type='int', default=5000)
(opts, _) = options.parse_args()
# </editor-fold>
# <editor-fold desc="Add paths beyond parent dir">
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
# </editor-fold>

from digtextsimilaritysearch.vectorizer.sentence_vectorizer \
    import SentenceVectorizer


# Doc I/O Funcs
def get_doc_count(file_loc):
    i_count = 0
    e_count = 0
    d_count = 0
    with open(file_loc, 'r') as fp:
        for line in fp:
            i_count += 1
            doc = json.loads(line)
            content = doc['lexisnexis']['doc_description']
            if content == '':
                e_count += 1
            if content == 'DELETED_STORY':
                d_count += 1

    print('Found {} empty stories'.format(e_count))
    print('Found {} deleted stories'.format(d_count))
    return i_count


def gen_docs(file_loc):
    with open(file_loc, 'r') as ifl:
        for line in ifl:
            yield json.loads(line)


def gen_sent_ids(doc_generator, b_size=100):
    batched_text = list()
    batched_ids = list()
    for ii, doc in enumerate(doc_generator, start=1):
        # ss = None
        # try:
        #     ss = doc['split_sentences']
        # except Exception as e:
        #     print(e)
        #     print(ii)
        #     print('This doc dont got no splits!')
        #     print(doc)
        # if not ss:
        #     pass
        # else:
        # content = None
        # try:
        # except:
        #     pass
        content = doc['lexisnexis']['doc_description']
        if content and not content == '' and not content == 'DELETED_STORY':
            text = list()
            text.append(doc['lexisnexis']['doc_title'])
            text.extend(doc['split_sentences'])

            doc_id = doc['doc_id']
            base_sent_id = np.int64(doc_id + '0000')
            sent_ids = list()
            for jj, _ in enumerate(text):
                sent_ids.append(base_sent_id + jj)
            sent_ids = np.vstack(sent_ids).astype(np.int64)
            assert sent_ids.shape[0] == len(text)
            # yield text, sent_ids  # Yields one doc at a time

            batched_text.extend(text)
            batched_ids.append(sent_ids)
            if ii % b_size == 0:
                batched_ids = np.vstack(batched_ids).astype(np.int64)
                print(len(batched_text) == batched_ids.shape[0])
                yield batched_text, batched_ids
                batched_text = list()
                batched_ids = list()
    batched_ids = np.vstack(batched_ids).astype(np.int64)
    yield batched_text, batched_ids


# Extend staticmethod to save lists of results
save = SentenceVectorizer.save_with_ids  # Handles file extensions internally


def consolidate_and_save(fp, id_sb, emb_sb, sent_sb):
    if isinstance(id_sb, list):
        id_sb = np.vstack(id_sb).astype(np.int64)
    if isinstance(emb_sb, list):
        emb_sb = np.vstack(emb_sb).astype(np.float32)

    assert len(id_sb) == len(emb_sb), 'len(ids)={}  len(emb)={}'.format(len(id_sb), len(emb_sb))

    save(file_path=fp, sent_ids=id_sb, embeddings=emb_sb, sentences=sent_sb)


t_start = time()


# Paths
split_news_filepath = opts.input_file
assert os.path.exists(split_news_filepath), \
    'Input file does not exist: {}'.format(split_news_filepath)
output_dir = opts.output_dir
assert os.path.isdir(output_dir), \
    'Output dir does not exist: {}'.format(output_dir)
output_filepath = None  # Defined in loop


# Init
sv = SentenceVectorizer()
doc_loader = gen_docs(file_loc=split_news_filepath)
docs_per_npz = opts.batch_size
sents_and_ids = gen_sent_ids(doc_generator=doc_loader, b_size=docs_per_npz)

t_init = time()
print('Initialized in {:0.2f}s'.format(t_init - t_start))
timestamps = list()
timestamps.append(time() - t_init)


# Run it
# report_interval = 2
n_docs = get_doc_count(split_news_filepath)
print('Processing {} docs...\n'.format(n_docs))
# id_superbatch = list()
# emb_superbatch = list()
# sent_superbatch = list()
for i, (sent_batch, id_batch) in enumerate(sents_and_ids):
    t_loop = time()

    # Process
    # id_superbatch.append(id_batch)
    # emb_superbatch.append(sv.make_vectors(sent_batch))
    # sent_superbatch.extend(sent_batch)

    # Save interval depreciated
    # # Save docs every b_size * save_intvl (default 1000)
    # if i % save_intvl == 0:
    #     id_superbatch = list()
    #     emb_superbatch = list()
    #     sent_superbatch = list()

    emb_batch = sv.make_vectors(sent_batch)
    timestamps.append(time() - t_loop)
    # Naw just always report
    # # TODO: Parse report interval
    # # if i % report_interval == 0:
    m, s = divmod(sum(timestamps[1:]), 60)
    print('  {:5d}/{} documents vectorized in {}m{:0.2f}s'
          ''.format((i+1)*docs_per_npz, n_docs, int(m), s))

    if opts.output_filename:
        output_filename = opts.output_filename.split('.')[0] + str(i*docs_per_npz)
        output_filepath = os.path.join(output_dir, output_filename)
    else:
        output_filename = str(split_news_filepath.split('/')[-1]).split('.')[0]
        output_filename = opts.seed + output_filename + str(i*docs_per_npz)
        output_filepath = os.path.join(output_dir, output_filename)

    emb_batch = np.vstack(emb_batch).astype(np.float32)
    print('Final {}'.format(emb_batch.shape[0] == id_batch.shape[0]))
    consolidate_and_save(fp=output_filepath, id_sb=id_batch,
                         emb_sb=emb_batch, sent_sb=sent_batch)
    # sv.save_with_ids(file_path=output_filepath, sent_ids=id_batch,
    #                  embeddings=emb_batch, sentences=sent_batch)


# Save every time, so no more "last part"
# # Save last part
# print('\n')
# print(len(sent_superbatch))
# if len(sent_superbatch):
#     consolidate_and_save(fp=output_filepath, id_sb=id_superbatch,
#                          emb_sb=emb_superbatch, sent_sb=sent_superbatch)
#     print('saved!')


mf, sf = divmod(time() - t_start, 60)
print('\nProcess completed in {}m{:0.2f}s'.format(int(mf), sf))
