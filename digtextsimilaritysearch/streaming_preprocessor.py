import os
import re
import json
import datetime
import numpy as np
from time import time
from optparse import OptionParser
from indexer.IVF_disk_index_handler import DiskBuilderIVF
from vectorizer.sentence_vectorizer import SentenceVectorizer
from process_documents.document_processor import DocumentProcessor
# <editor-fold desc="Parse Command Line Options">
cwd = os.path.abspath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
prog_file_path = os.path.join(cwd, 'progress.txt')
base_index_dir = os.path.abspath(os.path.join(cwd, '../saved_indexes/IVF16K_indexes/'
                                                   'emptyTrainedIVF16384.index'))
options = OptionParser()
options.add_option('-i', '--input_dir')
options.add_option('-o', '--output_dir')
options.add_option('-p', '--progress_file', default=prog_file_path)
options.add_option('-b', '--base_index_path', default=base_index_dir)
options.add_option('-m', '--m_per_batch', type='int', default=250000)
options.add_option('-r', '--report', action='store_true', default=False)
options.add_option('-d', '--delete_tmp_files', action='store_true', default=False)
(opts, _) = options.parse_args()
# </editor-fold>


"""
Options:
    -i  Path to raw news dir
    -o  Path to dir for writing merged, on-disk faiss index shard
    -p  File to keep track of news that has already been processed
    -b  Path to empty, pre-trained faiss index
    -m  Minimum number of sentences/vectors per .npz/.index (default 250000)
    -r  Bool to toggle prints
    -d  Bool to delete intermediate .npz/.index files
"""


# Funcs
def aggregate_docs(file_path, b_size=250000):
    doc_count = 0
    line_count = 0
    with open(file_path, 'r') as jl:
        for doc in jl:
            doc_count += 1
            line_count += len(json.loads(doc)['split_sentences']) + 1
    print('* Found {} lines in {} documents\n'
          '* {} batches will be processed\n'
          ''.format(line_count, doc_count,
                    divmod(line_count, b_size)[0] + 1))

    batched_text = list()
    batched_ids = list()
    with open(file_path, 'r') as jl:
        for doc in jl:
            document = json.loads(doc)

            content = document['lexisnexis']['doc_description']
            if content and not content == '' and not content == 'DELETED_STORY':
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


def check_unique(path, i=0):
    if os.path.exists(path):
        print('\nWarning: File already exists  {}'.format(path))
        path = path.split('.')
        path = path[0] + '_{}.'.format(i) + path[-1]
        print('         Testing new path  {}\n'.format(path))
        i += 1
        check_unique(path=path, i=i)
    return path


def clear(tmp_dir_path):
    for (tmp_dir, _, tmp_files) in os.walk(tmp_dir_path):
        for file in tmp_files:
            os.remove(os.path.join(tmp_dir, file))
    os.rmdir(tmp_dir_path)


# Track progress
preprocessed_news = list()
if os.path.exists(opts.progress_file):
    with open(opts.progress_file, 'r') as f:
        for line in f:
            preprocessed_news.append(str(line))

raw_news = list()
for (dir_path, _, file_list) in os.walk(opts.input_dir):
    for f in file_list:
        if f.endswith('.jl'):
            raw_news.append(os.path.join(dir_path, f))
    break

files_to_process = list()
for f in raw_news:
    if f not in preprocessed_news:
        files_to_process.append(f)
files_to_process.sort(reverse=True)
file_to_process = files_to_process[:1]   # Only preprocesses one news.jl


# Init paths
input_dir = opts.input_dir
date_today = str(datetime.date.today())
if date_today in file_to_process[0]:
    date = date_today
else:
    seed = str('\d{4}[-/]\d{2}[-/]\d{2}')
    date = re.search(seed, file_to_process[0]).group()

daily_dir = os.path.join(input_dir, date)
npz_dir = os.path.join(daily_dir, 'npzs')
subidx_dir = os.path.join(daily_dir, 'subindexes')
if not os.path.isdir(daily_dir):
    os.mkdir(daily_dir)
if not os.path.isdir(npz_dir):
    os.mkdir(npz_dir)
if not os.path.isdir(subidx_dir):
    os.mkdir(subidx_dir)


# Init DocumentProcessor
idx_bdr = DiskBuilderIVF(path_to_empty_index=opts.base_index_path)
sv = SentenceVectorizer()
dp = DocumentProcessor(indexer=None, index_builder=idx_bdr,
                       vectorizer=sv, storage_adapter=None)


# Preprocessing
def main():
    for raw_jl in file_to_process:
        if opts.report:
            print('\nProcessing: {}'.format(raw_jl))

        t_start = time()
        doc_batch_gen = aggregate_docs(file_path=raw_jl, b_size=opts.m_per_batch)
        for i, (batched_sents, batched_ids) in enumerate(doc_batch_gen):

            # Vectorize
            batched_embs = next(dp.vectorizer.make_vectors(batched_sents,
                                                           yield_vectors=True))

            # Save to .npz
            npz = str(raw_jl.split('/')[-1]).replace('.jl', '_{:03d}.npz'.format(i))
            npz_path = os.path.join(npz_dir, npz)
            npz_path = check_unique(path=npz_path)
            dp.vectorizer.save_with_ids(file_path=npz_path, embeddings=batched_embs,
                                        sentences=batched_sents, sent_ids=batched_ids)

            # Make faiss subindex
            subidx = 'subidx_' + str(npz_path.split('/')[-1]).replace('.npz', '.index')
            subidx_path = os.path.join(subidx_dir, subidx)
            subidx_path = check_unique(path=subidx_path)
            dp.index_docs_on_disk(path_to_npz=npz_path, path_to_invlist=subidx_path)

            if opts.report:
                mp, sp = divmod(time() - t_start, 60)
                print('  Completed doc batch: {:3d}          '
                      '  Time: {:3d}m{:0.2f}s'.format(i, int(mp), sp))

        # Merge
        t_merge = time()
        merged_ivfs = date + '_mergedIVF16K.ivfdata'
        merged_ivfs = os.path.join(opts.output_dir, merged_ivfs)
        merged_ivfs = check_unique(path=merged_ivfs)

        merged_index = date + '_populatedIVF16K.index'
        merged_index = os.path.join(opts.output_dir, merged_index)
        merged_index = check_unique(path=merged_index)

        if opts.report:
            print('\n  Merging {} on-disk'.format(merged_index.split('/')[-1]))
        dp.build_index_on_disk(merged_ivfs_path=merged_ivfs,
                               merged_index_path=merged_index)

        if opts.report:
            mm, sm = divmod(time() - t_merge, 60)
            print('\n  Merged subindexes in: {:3d}m{:0.2f}s'.format(int(mm), sm))

        # Record progress
        with open(opts.progress_file, 'a') as p:
            p.write(raw_jl + '\n')

        if opts.delete_tmp_files:
            clear(npz_dir)
            clear(subidx_dir)


if __name__ == '__main__':
    main()