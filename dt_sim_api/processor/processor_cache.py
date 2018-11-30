import collections
import functools


class Memoized(object):
    """
    Decorator:
        Caches a function's return value each time it is called.
        If called later with the same arguments, the cached value
        is returned (not reevaluated).
    """
    def __init__(self, func):
        self.func = func
        self.cache = dict()

    def __call__(self, *args):
        if not isinstance(args, collections.Hashable):
            # Not cacheable. (for instance: a list)
            # better to not cache (may blow up)
            return self.func(*args)
        if args in self.cache:
            return self.cache[args]
        else:
            value = self.func(*args)
            self.cache[args] = value
            return value

    def __repr__(self):
        """ Returns the function's docstring. """
        return self.func.__doc__

    def __get__(self, obj, objtype):
        """ Support instance methods. """
        return functools.partial(self.__call__, obj)
