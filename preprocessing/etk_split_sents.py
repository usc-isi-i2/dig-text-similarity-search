import os
import json
from time import time
from optparse import OptionParser
# <editor-fold desc="Parse Params">
param_parser = OptionParser()
param_parser.add_option('-i', '--input', dest='input_file',
                        help="Specify input file with '-i filename.jl'")
param_parser.add_option('-d', '--raw_dir', dest='raw_news_dir',
                        default='/lfs1/dig_text_sim/raw_news/')
param_parser.add_option('-s', '--split_dir', dest='split_news_dir',
                        default='/lfs1/dig_text_sim/split_news/')
param_parser.add_option('-r', '--report_intvl', dest='report_interval',
                        type='int', default=1000)
param_parser.add_option('-o', '--openblas', dest='set_openblas', default=True)
(opts, args) = param_parser.parse_args()
# </editor-fold>
# <editor-fold desc="Set OpenBLAS num threads">
if opts.set_openblas:
    os.environ['OPENBLAS_NUM_THREADS'] = '2'
    print('OpenBLAS n_threads: {}'.format(os.environ['OPENBLAS_NUM_THREADS']))
# </editor-fold>
from etk.etk import ETK
from etk.document import Document
from etk.etk_module import ETKModule
from etk.extractors.sentence_extractor import SentenceExtractor


class SentenceSplitter(ETKModule):
    def __init__(self, etk: ETK):
        ETKModule.__init__(self, etk)
        self.sentence_extractor = SentenceExtractor(name="dig-text-similarity-search")

    def process_document(self, doc: Document):
        text_to_split = doc.select_segments('lexisnexis.doc_description')
        split_sentences = doc.extract(self.sentence_extractor, text_to_split[0])
        doc.store(split_sentences, "split_sentences")

        return list()


def get_doc_count(file_loc):
    i_count = 0
    with open(file_loc, 'r') as fp:
        for _ in fp:
            i_count += 1
            pass
    return i_count


def gen_doc(file_loc):
    with open(file_loc, 'r') as fp:
        for line in fp:
            yield json.loads(line)


def gen_split(doc_gen):
    etk = ETK(modules=SentenceSplitter)

    for d in doc_gen:
        doc = etk.create_document(d)
        yield etk.process_ems(doc)


def add_doc(file_loc, split_gen):
    with open(file_loc, 'a') as sfp:
        for doc in split_gen:
            sfp.write(json.dumps(doc.cdr_document) + '\n')


# Dir Paths
daily_news_dir = opts.raw_news_dir
split_news_dir = opts.split_news_dir
assert os.path.isdir(split_news_dir), 'Try: mkdir {}'.format(split_news_dir)

# Files
raw_file = opts.input_file
day_of_news = os.path.join(daily_news_dir, raw_file)
assert os.path.isfile(day_of_news), 'Must pass target filename as argument'
write_name = 'split_' + raw_file.split('/')[-1]
day_of_splits = os.path.join(split_news_dir, write_name)
assert not os.path.isfile(day_of_splits), 'Script requires a clean run'


# Print n_docs
n_docs = get_doc_count(day_of_news)
print('\nFound {} documents in {}\n'.format(n_docs, day_of_news))
rep_invl = opts.report_interval

# Run it
t_0, t_1 = time(), time()
doc_generator = gen_doc(day_of_news)
split_generator = gen_split(doc_generator)
time_stamps = list()

for i, split_doc in enumerate(split_generator):
    add_doc(day_of_splits, split_doc)

    if i % rep_invl == 0:
        m, s = divmod(time()-t_0, 60)
        print(' {:7d} Docs processed in {}m{:0.2f}s'.format(i, int(m), s))
        t_diff = time() - t_1
        time_stamps.append(t_diff)
        t_1 = time()
        print('       * Avg time per doc: {:0.4f}s\n'.format(sum(time_stamps)/(i+1)))

m, s = divmod(time()-t_0, 60)
print('\nProcess completed in {}m{:0.2f}s'.format(int(m), s))
