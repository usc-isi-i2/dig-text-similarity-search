import unittest
from storage.memory_storage import MemoryStorage


class TestDocumentProcessor(unittest.TestCase):
    def setUp(self):
        self.ms = MemoryStorage()

    def test_create_table(self):
        self.ms.create_table('test_1')
        self.ms.create_table('test_2')
        tables = self.ms.tables()
        self.assertTrue(len(tables) == 2)

    def test_insert_record(self):
        self.ms.create_table('test_1')
        self.ms.insert_record('3', {'col_1': 'value_1'}, 'test_1')
        r = self.ms.get_record('3', 'test_1')
        self.assertEqual({'col_1': 'value_1'}, r)
