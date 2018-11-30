import pickle

__all__ = ["memoize"]


def memoize(func, limit: int = None):
    if isinstance(func, float):
        def memoize_wrapper(f):
            return memoize(f, func)

        return memoize_wrapper

    cache = dict()
    queue = list()

    def memoize_wrapper(*args, **kwargs):
        key = pickle.dumps((args, kwargs))
        try:
            queue.append(queue.pop(queue.index(key)))
        except ValueError:
            cache[key] = func(*args, **kwargs)
            queue.append(key)
            if limit is not None and len(queue) > limit:
                del cache[queue.pop(0)]

        return cache[key]

    memoize_wrapper._cache = cache
    memoize_wrapper._queue = queue
    memoize_wrapper._limit = limit
    memoize_wrapper._function = func
    memoize_wrapper.__name__ = func.__name__

    return memoize_wrapper
