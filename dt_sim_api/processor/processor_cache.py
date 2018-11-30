# import pickle
#
# __all__ = ["memoize"]
#
#
# def memoize(func, limit: int = None):
#     if isinstance(func, float):
#         def memoize_wrapper(f):
#             return memoize(f, func)
#
#         return memoize_wrapper
#
#     cache = dict()
#     queue = list()
#
#     def memoize_wrapper(*args, **kwargs):
#         key = pickle.dumps((args, kwargs))
#         try:
#             queue.append(queue.pop(queue.index(key)))
#         except ValueError:
#             cache[key] = func(*args, **kwargs)
#             queue.append(key)
#             if limit is not None and len(queue) > limit:
#                 del cache[queue.pop(0)]
#
#         return cache[key]
#
#     memoize_wrapper._cache = cache
#     memoize_wrapper._queue = queue
#     memoize_wrapper._limit = limit
#     memoize_wrapper._function = func
#     memoize_wrapper.__name__ = func.__name__
#
#     return memoize_wrapper


import collections
import functools


class Memoized(object):
    """
    Decorator. Caches a function's return value each time it is called.
       If called later with the same arguments, the cached value is returned
       (not reevaluated).
    """
    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args):
        if not isinstance(args, collections.Hashable):
            # Not cacheable. a list, for instance.
            # better to not cache than blow up.
            return self.func(*args)
        if args in self.cache:
            return self.cache[args]
        else:
            value = self.func(*args)
            self.cache[args] = value
            return value

    def __repr__(self):
        """
        :return: Return the function's docstring.
        """
        return self.func.__doc__

    def __get__(self, obj, objtype):
        """ Support instance methods. """
        return functools.partial(self.__call__, obj)
