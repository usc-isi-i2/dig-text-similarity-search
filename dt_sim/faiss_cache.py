import pickle
from threading import RLock

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

    cache = dict()
    queue = list()
    lock = RLock()

    def faiss_cache_wrapper(*args, **kwargs):
        key = pickle.dumps((args[1:], kwargs))  # Skip Faiss index.self arg
        with lock:
            try:
                queue.append(queue.pop(queue.index(key)))
            except ValueError:
                cache[key] = cacheable_func(*args, **kwargs)
                queue.append(key)
                if limit and len(queue) > limit:
                    del cache[queue.pop(0)]

        return cache[key]

    faiss_cache_wrapper._cache = cache
    faiss_cache_wrapper._queue = queue
    faiss_cache_wrapper._limit = limit
    faiss_cache_wrapper._function = cacheable_func
    faiss_cache_wrapper.__name__ = cacheable_func.__name__

    return faiss_cache_wrapper

