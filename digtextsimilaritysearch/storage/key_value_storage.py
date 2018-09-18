class KeyValueStorage(object):
    def __init__(self):
        pass

    def get_record(self, record_id, table_name):
        raise NotImplementedError

    def insert_record(self, record_id, record, table_name):
        raise NotImplementedError

    def create_table(self, table_name):
        raise NotImplementedError

    def tables(self):
        raise NotImplementedError

    def insert_records_batch(self, records, table_name, doc_type):
        raise NotImplementedError
