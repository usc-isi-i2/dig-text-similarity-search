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

    def __init__(self, host, size=5, **kwargs):
        KeyValueStorage.__init__(self)
        self._conn_pool = happybase.ConnectionPool(size=size, host=host, **kwargs)
        self._timeout = 6000000
        self._tables = {}

    def __del__(self):
        try:
            with self._conn_pool.connection(timeout=self._timeout) as _conn:
                _conn.close()
        except Exception as e:
            print('Exception: {}, while closing connection pool'.format(e))

    def tables(self):
        try:
            with self._conn_pool.connection(timeout=self._timeout) as _conn:
                return _conn.client.getTableNames()
        except Exception as e:
            print('Exception: {}, while getting table names'.format(e))

    def create_table(self, table_name, family_name='dig'):
        if not bytes(table_name, encoding='utf-8') in self.tables():
            try:
                with self._conn_pool.connection(timeout=self._timeout) as _conn:
                    _conn.create_table(table_name, {family_name: dict()})
            except Exception as e:
                print('Exception: {}, while creating table: {}'.format(e, table_name))

    def get_record(self, record_id, table_name,
                   column_names=(_SENTENCE_ID, _SENTENCE_TEXT),
                   column_family='dig'):
        try:
            with self._conn_pool.connection(timeout=self._timeout) as _conn:
                record = self.get_table(table_name).row(record_id)
                if record:
                    result = {}
                    for column_name in column_names:
                        fam_col = '{}:{}'.format(column_family, column_name).encode('utf-8')
                        result[column_name] = record.get(fam_col, '').decode('utf-8')
                    return result
        except Exception as e:
            print('Exception: {}, while retrieving record: {}, '
                  'from table: {}'.format(e, record_id, table_name))

        return None

    def get_table(self, table_name):
        if table_name not in self._tables:
            try:
                with self._conn_pool.connection(timeout=self._timeout) as _conn:
                    self._tables[table_name] = _conn.table(table_name)
            except Exception as e:
                print('Exception:{}, while retrieving table: {}'.format(e, table_name))
                return None
        return self._tables[table_name]

    def insert_record_value(self, record_id, value, table_name, column_family, column_name):
        try:
            with self._conn_pool.connection(timeout=self._timeout) as _conn:
                table = self.get_table(table_name)
                if table:
                    return table.put(record_id, {'{}:{}'.format(column_family, column_name): value})
                raise Exception('Table: {} not found'.format(table_name))
        except Exception as e:
            pass

    def insert_record(self, record_id, record, table_name):
        try:
            with self._conn_pool.connection(timeout=self._timeout) as _conn:
                table = self.get_table(table_name)
                if table:
                    return table.put(record_id, record)
                raise Exception('Table: {} not found'.format(table_name))
        except Exception as e:
            print('Exception: {}, while writing record (id:{}, val:{}) '
                  'to table: {}'.format(e, record_id, record, table_name))

    def insert_records_batch(self, records, table_name):
        """
        Adds records into hbase in batch mode
        :param records: list of records to be inserted, each record a tuple (id, data)
        :param table_name: table in hbase where records will be shipped to
        :return: exception in case of failure
        """
        try:
            with self._conn_pool.connection(timeout=self._timeout) as _conn:
                table = self.get_table(table_name)
                batch = table.batch()
                for record in records:
                    batch.put(record[0], record[1])
                batch.send()
        except Exception as e:
            print('Exception: {}, while writing batch of records '
                  'to table: {}'.format(e, table_name))

    def delete_table(self, table_name):
        try:
            with self._conn_pool.connection(timeout=self._timeout) as _conn:
                _conn.delete_table(table_name, disable=True)
        except Exception as e:
            print('Exception: {}, while deleting table: {}'.format(e, table_name))

    def __iter__(self, table_name):
        return self.__next__(table_name)

    def __next__(self, table_name):
        table = self.get_table(table_name)
        if table:
            for key, data in table.scan(filter=b'FirstKeyOnlyFilter()'):
                yield {key: data}
        else:
            raise Exception('table: {} not found'.format(table_name))
