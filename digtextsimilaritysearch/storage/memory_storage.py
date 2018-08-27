from digtextsimilaritysearch.storage.key_value_storage import KeyValueStorage


class MemoryStorage(KeyValueStorage):
    def __init__(self):
        KeyValueStorage.__init__(self)
        self._db = dict()

    def get_record(self, record_id, table_name):
        if table_name in self._db:
            return self._db[table_name].get(record_id, None)
        return None

    def insert_record(self, record_id, record, table_name):
        self._db[table_name][record_id] = record

    def create_table(self, table_name):
        if table_name not in self._db:
            self._db[table_name] = dict()

    def tables(self):
        return list(self._db)
