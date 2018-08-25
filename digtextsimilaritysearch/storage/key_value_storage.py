class KeyValueStorage(object):
    def __init__(self):
        pass

    def get_record(self, record_id):
        raise NotImplementedError

    def insert_record(self, record_id, record):
        raise NotImplementedError
