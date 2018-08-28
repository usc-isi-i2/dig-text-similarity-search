import unittest
from digtextsimilaritysearch.storage.hbase_adapter import HBaseAdapter


class TestDocumentProcessor(unittest.TestCase):
    def setUp(self):
        print('Ensure Hbase is up and running...')
        self.hb = HBaseAdapter('localhost')

    def test_create_table(self):
        self.hb.create_table('test_2')
        self.hb.create_table('test_1')
        self.hb.create_table('test_1')
        tables = list()
        for x in self.hb.tables():
            if x.decode('utf-8').startswith('test_'):
                tables.append(x.decode('utf-8'))

        self.assertTrue(len(tables) == 2)

    def test_insert_record(self):
        self.hb.create_table('test_1')
        self.hb.insert_record('3', {'dig:col_1': 'value_1'}, 'test_1')
        r = self.hb.get_record('3', 'test_1', column_names=['col_1'])
        self.assertEqual({'col_1': 'value_1'}, r)

    def tearDown(self):
        self.hb.delete_table('test_1')
        self.hb.delete_table('test_2')
