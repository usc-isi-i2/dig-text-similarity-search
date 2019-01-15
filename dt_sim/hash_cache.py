import pickle
import functools


__all__ = ['PickleMemo']


class PickleMemo(object):
    """
    Decorator:
        Caches a function's return value each time it is called.
        If called later with the same arguments, the cached value
        is returned (not reevaluated).

        Works with kwargs and Faiss index.search()

        Holds maxsize results. Drops the Least Recently Used result.
    """
    def __init__(self, func, maxsize: int = 128):
        self.func = func
        self.cache = dict()
        self.queue = list()
        self.maxsize = maxsize

    def __call__(self, *args, **kwargs):
        key = pickle.dumps((args[1:], kwargs))      # Only use embedding arg
        if key in self.cache:
            self.queue.append(self.queue.pop(self.queue.index(key)))
            return self.cache[key]
        else:
            value = self.func(*args, **kwargs)
            self.queue.append(key)
            if len(self.queue) > self.maxsize:
                del self.cache[self.queue.pop(0)]
            self.cache[key] = value
            return value

    def __repr__(self):
        """ Returns the function's docstring. """
        return self.func.__doc__

    def __get__(self, obj, objtype):
        """ Support instance methods. """
        return functools.partial(self.__call__, obj)
