import unittest
import json
from digtextsimilaritysearch.process_documents.document_processor import DocumentProcessor
from vectorizer.sentence_vectorizer import SentenceVectorizer
from indexer.faiss_indexer import FaissIndexer
from storage.memory_storage import MemoryStorage


class TestDocumentProcessor(unittest.TestCase):
    def setUp(self):
        # self.docs = [json.loads(line) for line in open('digtextsimilaritysearch/unit_tests/resources/news_sample.jl')]
        self.docs = [json.loads(line) for line in open('resources/news_sample.jl')]
        self.fi = FaissIndexer()
        self.sv = SentenceVectorizer()
        self.ms = MemoryStorage()

        self.dp = DocumentProcessor(self.fi, self.sv, self.ms, table_name='test_1')
        sample_docs = self.docs[0]
        self.dp.index_documents(sample_docs)

    def test_preprocess_documents(self):
        dp_1 = DocumentProcessor(None, None, None, None)
        sentences = dp_1.preprocess_documents(self.docs)
        self.assertTrue(len(sentences) == 1076)
        # last three
        sids = [s[0] for s in sentences[1073:1076]]
        self.assertEqual(sids, ['27c656a975d5542c3d22a2a72ed00e482350898505d4f86834e4abe48228cc0c_0',
                                '27c656a975d5542c3d22a2a72ed00e482350898505d4f86834e4abe48228cc0c_1',
                                '27c656a975d5542c3d22a2a72ed00e482350898505d4f86834e4abe48228cc0c_2'])

    def test_vector_creation(self):
        r = self.ms.get_record('37', 'test_1')
        self.assertEqual(r['dig:sentence_id'], 'e36da14c6246c2f737925284002249e91583f94c06a3f95a56cb8733364ec696_37')
        self.assertEqual(r['dig:sentence_text'], '+    Return on Equity 11.2%')

    def test_query(self):
        r = self.dp.query_text('return on equity', k=3)
        expected_r = [
            {'doc_id': 'e36da14c6246c2f737925284002249e91583f94c06a3f95a56cb8733364ec696', 'score': 0.25193464756011963,
             'sentence': '+    Return on Equity 11.2%'},
            {'doc_id': 'e36da14c6246c2f737925284002249e91583f94c06a3f95a56cb8733364ec696', 'score': 0.5006510615348816,
             'sentence': 'JPY2,013, including a capital gain of JPY736 and dividend reinvested of JPY276.'},
            {'doc_id': 'e36da14c6246c2f737925284002249e91583f94c06a3f95a56cb8733364ec696', 'score': 0.5174432992935181,
             'sentence': '[3.1%].+    Return on Capital Employed 15.5%'}]
        self.assertEqual(r, expected_r)
