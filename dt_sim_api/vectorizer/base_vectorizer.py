class BaseVectorizer(object):
    def __init__(self):
        self.large_USE = False
        raise NotImplementedError

    def make_vectors(self, text):
        raise NotImplementedError
