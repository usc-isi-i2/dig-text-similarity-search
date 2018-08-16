import unittest
import json
from digtextsimilaritysearch.process_documents.document_processor import DocumentProcessor


class TestDocumentProcessor(unittest.TestCase):
    def setUp(self):
        self.docs = [json.loads(line) for line in open('digtextsimilaritysearch/unit_tests/resources/news_sample.jl')]
        self.dp = DocumentProcessor(None, None, None, None)

    def test_preprocess_documents(self):
        sentences = self.dp.preprocess_documents(self.docs)
        self.assertTrue(len(sentences) == 1076)
        # last three
        sids = [s[0] for s in sentences[1073:1076]]
        self.assertEqual(sids, ['27c656a975d5542c3d22a2a72ed00e482350898505d4f86834e4abe48228cc0c_0',
                                '27c656a975d5542c3d22a2a72ed00e482350898505d4f86834e4abe48228cc0c_1',
                                '27c656a975d5542c3d22a2a72ed00e482350898505d4f86834e4abe48228cc0c_2'])
