import unittest
from storage.hbase_adapter import HBaseAdapter


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

    def test_batch_insert(self):
        self.hb.create_table('test_1')
        doc_list = list()

        for i in range(0, 10):
            doc_list.append((str(i), {'dig:col_1': 'this_{}'.format(i)}))
        self.hb.insert_records_batch(doc_list, 'test_1')
        expected_r_1 = {'col_1': 'this_4'}
        expected_r_2 = {'col_1': 'this_9'}
        self.assertEqual(expected_r_1, self.hb.get_record('4', 'test_1', column_names=['col_1']))
        self.assertEqual(expected_r_2, self.hb.get_record('9', 'test_1', column_names=['col_1']))

    def tearDown(self):
        self.hb.delete_table('test_1')
        self.hb.delete_table('test_2')
