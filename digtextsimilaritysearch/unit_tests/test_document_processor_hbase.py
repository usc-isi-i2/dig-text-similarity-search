import unittest
import json
from process_documents.document_processor import DocumentProcessor
from vectorizer.sentence_vectorizer import SentenceVectorizer
from indexer.faiss_indexer import FaissIndexer
from storage.hbase_adapter import HBaseAdapter


class TestDocumentProcessor(unittest.TestCase):
    def setUp(self):
        self.docs = [json.loads(line) for line in open('unit_tests/resources/news_sample.jl')]
        self.fi = FaissIndexer()
        self.sv = SentenceVectorizer()
        self.ms = HBaseAdapter('localhost')

        self.dp = DocumentProcessor(self.fi, self.sv, self.ms, table_name='test_1')

    def test_index_documents(self):
        self.ms.create_table('test_1')
        sample_docs = self.docs[0:10]
        self.dp.index_documents(sample_docs, batch_mode=True, batch_size=5)
        expected_r = {'sentence_id': '0079a290a012bb3b57a1af34567fd807e6e07ecbf288da7daad72628d1de6c75_95',
                      'sentence_text': '1850 Miyanogi-cho Inage-ku Chiba City Chiba, 263-0054'}
        r = self.ms.get_record('1016', 'test_1')
        self.assertEqual(r, expected_r)

    def tearDown(self):
        self.ms.delete_table('test_1')
