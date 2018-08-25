class KeyValueStorage(object):
    def __init__(self):
        pass

    def get_record(self, record_id, table_name):
        raise NotImplementedError

    def insert_record(self, record_id, record, table_name):
        raise NotImplementedError
