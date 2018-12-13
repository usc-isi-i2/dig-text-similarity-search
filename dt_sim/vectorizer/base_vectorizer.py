class BaseVectorizer(object):
    def __init__(self):
        self.large_USE = False

    def make_vectors(self, text):
        raise NotImplementedError
