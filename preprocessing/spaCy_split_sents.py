import os
import json
import spacy
from time import time
from optparse import OptionParser


def get_doc_count(file_loc):
    i_count = 0
    with open(file_loc, 'r') as fp:
        for _ in fp:
            i_count += 1
    return i_count


def gen_doc(file_loc, b_size=16):
    docs = list()
    text = list()
    with open(file_loc, 'r') as fp:
        for line in fp:
            docs.append(json.loads(line))
            text.append(json.loads(line)['lexisnexis']['doc_description'])
            if len(docs) >= b_size:
                yield docs, text
                docs = list()
                text = list()


def gen_split(doc_gen, mb_size=4, n_thr=2):
    parser = spacy.load("en_core_web_sm", disable=["tagger", "ner"])

    for docs, text in doc_gen:
        par_gen = parser.pipe(text, batch_size=mb_size, n_threads=n_thr)
        for jj, parsed_text in enumerate(par_gen):
            doc = dict(docs[jj])
            sents = list()
            for sent in parsed_text.sents:
                sents.append(sent.text.replace('\n', ' '))
            doc['split_sentences'] = sents
            yield doc


def add_doc(split_doc_gen, output_file, rep_invtl=1000):
    ii = 0
    t_0, t_1 = time(), time()
    time_stamps = list()
    with open(output_file, 'a') as sfp:
        for doc in split_doc_gen:
            sfp.write(json.dumps(doc) + '\n')

            if rep_invtl > 0 and ii % rep_invtl == 0:
                minu, sec = divmod(time() - t_0, 60)
                print(' {:7d} docs processed in {}m{:0.2f}s'
                      ''.format(ii, int(minu), sec))
                t_diff = time() - t_1
                t_1 = time()
                time_stamps.append(t_diff)
                print('       * Avg time per doc: {:0.6f}s'
                      '\n'.format(sum(time_stamps) / (ii + 1)))
            ii += 1


# Runtime Params
param_parser = OptionParser()
param_parser.add_option('-i', '--input', dest='input_file',
                        help="Specify input file with '-i filename.jl'")
param_parser.add_option('-b', '--batch', dest='batch_size',
                        type='int', default=16)
param_parser.add_option('-m', '--minibatch', dest='minibatch_size',
                        type='int', default=4)
param_parser.add_option('-t', '--threads', dest='n_threads',
                        type='int', default=2)
param_parser.add_option('-r', '--report_intvl', dest='report_interval',
                        type='int', default=1000)
(opts, args) = param_parser.parse_args()

raw_file = opts.input_file
docs_per_batch = opts.batch_size
docs_per_minibatch = opts.minibatch_size
usable_threads = opts.n_threads
report_interval = opts.report_interval

# Dir Paths
daily_news_dir = '/lfs1/dig_text_sim/raw_news/'
split_news_dir = '/lfs1/dig_text_sim/split_news/'
assert os.path.isdir(split_news_dir), 'Try: mkdir {}'.format(split_news_dir)

# Files
day_of_news = os.path.join(daily_news_dir, raw_file)
assert os.path.isfile(day_of_news), 'Must pass target filename as argument'
write_name = 'split_' + raw_file.split('/')[-1]
day_of_splits = os.path.join(split_news_dir, write_name)
assert not os.path.isfile(day_of_splits), 'Script requires a clean run'

# Print n_docs
n_docs = get_doc_count(day_of_news)
print('\nFound {} documents in {}\n'.format(n_docs, day_of_news))

# Generators
doc_generator = gen_doc(file_loc=day_of_news, b_size=docs_per_batch)
split_generator = gen_split(doc_gen=doc_generator,
                            mb_size=docs_per_minibatch, n_thr=usable_threads)

# Run it
t_start = time()
add_doc(split_doc_gen=split_generator,
        output_file=day_of_splits, rep_invtl=report_interval)

# Final Report
m, s = divmod(time()-t_start, 60)
print('\nProcess completed in {}m{:0.2f}s'.format(int(m), s))
