# <editor-fold desc="Imports">
import os
from optparse import OptionParser
# <editor-fold desc="Parse Command Line Options">
cwd = os.path.abspath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
prog_file_path = os.path.join(cwd, 'progress.txt')
relative_base_path = '../../saved_indexes/USE_lite_base_IVF16K.index'
base_index_path = os.path.abspath(os.path.join(cwd, relative_base_path))

options = OptionParser()
options.add_option('-i', '--input_file')
options.add_option('-t', '--num_threads')
options.add_option('-b', '--base_index_path', default=base_index_path)
options.add_option('-m', '--m_per_batch', type='int', default=512*128)
options.add_option('-n', '--n_per_minibatch', type='int', default=128)
options.add_option('-l', '--large', action='store_true', default=False)
options.add_option('-a', '--intra', type='int', default=8)  # TODO: docstring
options.add_option('-e', '--inter', type='int', default=2)
options.add_option('-r', '--report', action='store_true', default=False)
options.add_option('-v', '--verbose', action='store_true', default=False)
options.add_option('-s', '--skip', type='int', default=0)
(opts, _) = options.parse_args()
# </editor-fold>

if opts.num_threads:
    print('\nRestricting numpy to {} thread(s)\n'.format(opts.num_threads))
    os.environ['OPENBLAS_NUM_THREADS'] = opts.num_threads
    os.environ['NUMEXPR_NUM_THREADS'] = opts.num_threads
    os.environ['MKL_NUM_THREADS'] = opts.num_threads
    os.environ['OMP_NUM_THREADS'] = opts.num_threads

import re
import sys
import json
import datetime
import numpy as np
from time import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from dt_sim_api.vectorizer import SentenceVectorizer
from dt_sim_api.process_documents.document_processor import DocumentProcessor
# </editor-fold>


"""
To run this script:
    (from dig-text-similarity-search/)
    $ python preprocessing/streaming_preprocessor.py \ 
        -i {/path/to/split/sents.jl} -o {/path/to/write/shard.index} -r -d

Options:
    -i  Path to raw news dir
    -t  Option to set thread budget for numpy to reduce CPU resource consumption 
            Useful if other tasks are running 
    -b  Path to empty, pre-trained faiss index 
            (default ../saved_indexes/IVF16K_indexes/USE_lite_base_IVF16K.index)
    -m  Minimum number of sentences per batch 
            (default 512*128)
    -l  Bool to use large Universal Sentence Encoder
            (default False) 
    -r  Bool to toggle prints 
            (default False)
            
    -s  Development param: If preprocessing was interrupted after several 
            sub.index files were created, but before the on-disk shard was merged, 
            use -s <int:n_files_to_reuse> to reuse existing intermediate files. 
            * Note: Do NOT reuse partially created intermediate files
"""


# Funcs
def check_docs(file_path, b_size=512*128):
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
    n_batches = divmod(line_count, b_size)[0] + 1
    n_good_batches = divmod(good_sents, b_size)[0] + 1
    return doc_count, line_count, good_sents, junk_count, n_batches, n_good_batches


def aggregate_docs(file_path, b_size=512*128):
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


def check_unique(path, i=0):
    if os.path.exists(path):
        print('\nWarning: File already exists  {}'.format(path))
        path = path.split('.')
        path = path[0] + '_{}.'.format(i) + path[-1]
        print('         Testing new path  {}\n'.format(path))
        i += 1
        check_unique(path=path, i=i)
    return path


# Track progress
file_to_process = [opts.input_file]
if opts.report:
    print('Will process: {}\n'.format(file_to_process[0]))


# Init paths
date_today = str(datetime.date.today())
if date_today in file_to_process[0]:
    date = date_today
else:
    seed = str('\d{4}[-/]\d{2}[-/]\d{2}')
    date = re.search(seed, file_to_process[0]).group()

intermediate_dir = os.path.abspath(os.path.join(os.path.dirname(opts.input_file), 
                                                '../training_npzs/'))
daily_dir = os.path.join(intermediate_dir, date)
if not os.path.isdir(intermediate_dir):
    os.mkdir(intermediate_dir)
if not os.path.isdir(daily_dir):
    os.mkdir(daily_dir)

if opts.large:
    large_dir = '../digtextsimilaritysearch/vectorizer' \
                '/model/96e8f1d3d4d90ce86b2db128249eb8143a91db73'
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), large_dir))
else:
    model_dir = None    
    empty_idx = opts.base_index_path


# Init DocumentProcessor
# idx_bdr = DiskBuilderIVF(path_to_empty_index=opts.base_index_path)
sv = SentenceVectorizer(path_to_model=model_dir, intra=opts.intra, inter=opts.inter)
dp = DocumentProcessor(indexer=None, index_builder=None,
                       vectorizer=sv, storage_adapter=None)


# Preprocessing
def main():
    for raw_jl in file_to_process:
        if opts.report:
            print('\nReading file: {}'.format(raw_jl))

        line_counts = check_docs(file_path=raw_jl, b_size=opts.m_per_batch)
        (doc_count, line_count, good_sents, junk, n_batches, n_good_batches) = line_counts
        if opts.report:
            print('* Found {} good documents with {} lines and {} good sentences\n'
                  '* Will skip {} junk documents\n'
                  '* Processing {}:{} batches\n'
                  ''.format(doc_count, line_count, good_sents,
                            junk, n_good_batches, n_batches))

        t_start = time()
        doc_batch_gen = aggregate_docs(file_path=raw_jl, b_size=opts.m_per_batch)
        for i, (batched_sents, batched_ids) in enumerate(doc_batch_gen):
            t_0 = time()
            if opts.report:
                print('  Starting doc batch:  {:3d}'.format(i+1))

            npz = str(raw_jl.split('/')[-1]).replace('.jl', '_{:03d}_train.npz'.format(i))
            npz_path = os.path.join(daily_dir, npz)

            # TODO add skip
            if i < opts.skip and os.path.exists(npz_path):
                print('  File exists: {} \n'
                      '  Skipping'.format(npz_path))
            else:
                # Vectorize
                batched_embs = dp.vectorizer.make_vectors(batched_sents,
                                                          batch_size=opts.n_per_minibatch,
                                                          verbose=opts.verbose)
                t_vect = time()
                if opts.report:
                    print('  * Vectorized in {:6.2f}s'.format(t_vect - t_0))

                # Numpify
                if not isinstance(batched_embs, np.ndarray):
                    batched_embs = np.vstack(batched_embs).astype(np.float32)
                if not isinstance(batched_ids, np.ndarray):
                    try:
                        batched_ids = np.array(batched_ids, dtype=np.int64)
                    except ValueError:
                        print(batched_ids)
                        raise ValueError

                # Save npz for later
                npz_path = check_unique(path=npz_path)
                dp.vectorizer.save_with_ids(npz_path,
                                            embeddings=batched_embs,
                                            sentences=batched_sents,
                                            sent_ids=batched_ids,
                                            compressed=False)

                t_npz = time()
                if opts.report:
                    print('  * Saved .npz in {:6.2f}s'.format(t_npz - t_vect))

                # Clear graph
                del batched_embs, batched_sents, batched_ids
                dp.vectorizer.close_session()
                t_reset = time()
                if opts.report:
                    print('  * Cleared TF in {:6.2f}s'.format(t_reset - t_npz))

                # Restart TF session if necessary
                if i < n_batches - 1:
                    dp.vectorizer.start_session()
                    if opts.report:
                        print('  * Started TF in {:6.2f}s'.format(time() - t_reset))

                if opts.report:
                    mp, sp = divmod(time() - t_start, 60)
                    print('  Completed doc batch: {:3d}/{}      '
                          '  Total time passed: {:3d}m{:0.2f}s\n'
                          ''.format(i+1, n_good_batches, int(mp), sp))


if __name__ == '__main__':
    if len(file_to_process):
        main()
    else:
        print('Nothing to process.')
