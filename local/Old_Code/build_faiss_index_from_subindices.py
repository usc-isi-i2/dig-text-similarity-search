from optparse import OptionParser
from indexer.IVF_disk_index_handler import DiskBuilderIVF
from vectorizer.sentence_vectorizer import SentenceVectorizer
from process_documents.document_processor import DocumentProcessor
import glob
import os

cwd = os.path.abspath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
base_index_dir = os.path.abspath(os.path.join(cwd, '../saved_indexes/IVF16K_indexes/'
                                                   'emptyTrainedIVF16384.index'))
options = OptionParser()
options.add_option('-i', '--input_dir')
options.add_option('-o', '--output_path')
options.add_option('-n', '--index_name', default='index')
options.add_option('-b', '--base_index_path', default=base_index_dir)
(opts, _) = options.parse_args()

# Init DocumentProcessor
idx_bdr = DiskBuilderIVF(path_to_empty_index=opts.base_index_path)
sv = SentenceVectorizer()
dp = DocumentProcessor(indexer=None, index_builder=idx_bdr,
                       vectorizer=sv, storage_adapter=None)

output = opts.output_path
index_name = opts.index_name
ivf_path = '{}/{}.ivfdata'.format(output, index_name)
f_index_path = '{}/{}.index'.format(output, index_name)

subindex_paths = glob.glob('{}/*index'.format(opts.input_dir))

# Add paths
dp.index_builder.extend_invlist_paths(subindex_paths)

# Merge
dp.build_index_on_disk(merged_ivfs_path=ivf_path, merged_index_path=f_index_path)
