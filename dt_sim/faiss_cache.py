from collections import OrderedDict
from threading import RLock
from pickle import dumps

__all__ = ["faiss_cache"]


def faiss_cache(cacheable_func, limit: int = None):
    """
    Decorator: Caches a function's return value each time it is called.
        If called later with the same arguments, the cached value
        is returned (not reevaluated).

        Works with Faiss index.search()
        Holds maxsize results. Drops the Least Recently Used result.

    Note: Only pass positional args to wrapper (do NOT @faiss_cache(limit=int))
    """
    if isinstance(cacheable_func, int):
        def faiss_cache_wrapper(f):
            return faiss_cache(f, cacheable_func)

        return faiss_cache_wrapper

    cache_q = OrderedDict()
    lock = RLock()

    def faiss_cache_wrapper(*args, **kwargs):
        key = dumps((args[1:], kwargs))  # Skip Faiss index.self arg
        with lock:
            try:
                cache_q.move_to_end(key)
            except KeyError:
                cache_q[key] = cacheable_func(*args, **kwargs)
                if limit and len(cache_q) > limit:
                    cache_q.popitem(last=False)

        return cache_q[key]

    faiss_cache_wrapper._cache_q = cache_q
    faiss_cache_wrapper._limit = limit
    faiss_cache_wrapper._function = cacheable_func
    faiss_cache_wrapper.__name__ = cacheable_func.__name__

    return faiss_cache_wrapper
