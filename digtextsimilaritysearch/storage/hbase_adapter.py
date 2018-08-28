import happybase
from .key_value_storage import KeyValueStorage

_SENTENCE_ID = 'sentence_id'
_SENTENCE_TEXT = 'sentence_text'


class HBaseAdapter(KeyValueStorage):
    """
    Note: Need to increase timeout for thrift in hbase-site.xml
    <property>
        <name>hbase.thrift.server.socket.read.timeout</name>
        <value>6000000</value>
    </property>
    <property>
        <name>hbase.thrift.connection.max-idletime</name>
        <value>18000000</value>
    </property>
    """

    def __init__(self, host, **kwargs):
        KeyValueStorage.__init__(self)
        self._conn = happybase.Connection(host=host, timeout=6000000, **kwargs)
        self._tables = {}

    def __del__(self):
        try:
            self._conn.close()
        except:
            pass

    def tables(self):
        return self._conn.client.getTableNames()

    def create_table(self, table_name, family_name='dig'):
        if not bytes(table_name, encoding='utf-8') in self.tables():
            self._conn.create_table(table_name, {family_name: dict()})

    def get_record(self, record_id, table_name, column_names=[_SENTENCE_ID, _SENTENCE_TEXT], column_family='dig'):
        try:
            record = self.get_table(table_name).row(record_id)
            if record:
                result = {}
                for column_name in column_names:
                    fam_col = '{}:{}'.format(column_family, column_name).encode('utf-8')
                    result[column_name] = record.get(fam_col, '').decode('utf-8')
                return result
        except Exception as e:
            print('Exception: {}, while retrieving record: {}, from table: {}'.format(e, record_id, table_name))

        return None

    def get_table(self, table_name):
        try:
            if table_name not in self._tables:
                self._tables[table_name] = self._conn.table(table_name)
            return self._tables[table_name]
        except Exception as e:
            print('Exception:{}, while retrieving table: {}'.format(e, table_name))
        return None

    def insert_record_value(self, record_id, value, table_name, column_family, column_name):
        table = self.get_table(table_name)
        if table:
            return table.put(record_id, {'{}:{}'.format(column_family, column_name): value})
        raise Exception('table: {} not found'.format(table_name))

    def insert_record(self, record_id, record, table_name):
        table = self.get_table(table_name)
        if table:
            return table.put(record_id, record)
        raise Exception('table: {} not found'.format(table_name))

    def insert_records_batch(self, records, table_name):
        """
        Adds records into hbase in batch mode
        :param records: list of records to be inserted, each record a tuple (id, data)
        :param table_name: table in hbase where records will be shipped to
        :return: exception in case of failure
        """
        table = self.get_table(table_name)
        batch = table.batch()
        for record in records:
            batch.put(record[0], record[1])
        batch.send()

    def delete_table(self, table_name):
        try:
            self._conn.delete_table(table_name, disable=True)
        except:
            pass

    def __iter__(self, table_name):
        return self.__next__(table_name)

    def __next__(self, table_name):
        table = self.get_table(table_name)
        if table:
            for key, data in table.scan(filter=b'FirstKeyOnlyFilter()'):
                yield {key: data}
        else:
            raise Exception('table: {} not found'.format(table_name))
