from digtextsimilaritysearch.storage.key_value_storage import KeyValueStorage


class MemoryStorage(KeyValueStorage):
    def __init__(self):
        KeyValueStorage.__init__(self)
        self._records = dict

    def get_record(self, record_id):
        return self._records.get(record_id, None)

    def insert_record(self, record_id, record):
        self._records[record_id] = record
