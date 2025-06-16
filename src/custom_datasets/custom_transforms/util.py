import collections.abc
import numpy as np
import inspect


def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        assert len(x) == 2
        return tuple(x)
    return (x, x)


def get_rng_from_global():
    # on dataloader worker spawn:
    # various objects require to initialize their random number generators when workers are spawned
    # the naive solution is to initialize each rng with rng = np.random.default_rng(seed=info.seed)
    # but this raises the issue that when multiple objects require an rng initialization, all rngs
    # would be seeded with info.seed
    # solution: since the numpy global rng is seeded in the worker instantiation, sample seed for rng from global rng

    # on initialization:
    # np.random.default_rng() will not be affected by np.random.set_seed (i.e. by the global numpy random seed)
    # solution: sample a random integer from np.random.randint (which is affected by np.random.set_seed)
    return np.random.default_rng(seed=np.random.randint(np.iinfo(np.int32).max))


def optional_ctx(fn, ctx):
    # returns dict(ctx=ctx) if fn takes a ctx argument
    if ctx is not None and "ctx" in inspect.getfullargspec(fn).args:
        kwargs = dict(ctx=ctx)
    else:
        kwargs = {}
    return kwargs
