import numpy as np


import warnings
warnings.warn("This module is only for dirty testing!")


class dummymod:
    __name__ = "numpy_dispatch.dummymod"

    def __getattr__(self, attr):
        return getattr(np, attr)


class DummyArray(np.ndarray):
    def __array_module__(self, types):
        print(types)
        for type in types:
            if type is DummyArray:
                continue
            if type is np.ndarray:
                continue
            break
        else:
            return dummymod

        return NotImplemented


def array(*args, **kwargs):
    return np.array(*args, **kwargs).view(DummyArray)


class _FakeUfunc:
    def __init__(self, ufunc, arrself):
        self.ufunc = ufunc
        self.arrtypes = (type(arrself),)
        self.arrufunc = arrself.__array_ufunc__

    def prepare_args(self, args, nin=None):
        if nin is None:
            nin = self.ufunc.nin
        if len(args) != nin:
            raise RuntimeError("Fake array_module: only inputs can be arguments to ufuncs")

        new_args = []
        for arg in args:
            new = np.asarray(arg)
            new = self.arrtypes[0](new)
            new_args.append(new)

        return tuple(new_args)

    def __call__(self, *args, **kwargs):
        args = self.prepare_args(args)
        return self.arrufunc(self.ufunc, "__call__", self.arrtypes, args, kwargs)

    def reduce(self, *args, **kwargs):
        args = self.prepare_args(args, nin=1)
        return self.arrfunc(self.ufunc, "reduce", self.arrtypes, args, kwargs)

    def accumulate(self, *args, **kwargs):
        args = self.prepare_args(args, nin=1)
        return self.arrfunc(self.ufunc, "accumulate", self.arrtypes, args, kwargs)

    def outer(self, *args, **kwargs):
        args = self.prepare_args(args)
        return self.arrfunc(self.ufunc, "outer", self.arrtypes, args, kwargs)


class _BoundFunction:
    def __init__(self, function, arrself):
        self.func = function
        self.arrtypes = (type(arrself),)
        self.arrfunction = arrself.__array_function__

    def __call__(self, *args, **kwargs):
        return self.arrfunction(self.func, self.arrtypes, args, kwargs)


class _ArrayFunctionFallback:
    def __init__(self, name, arrself):
        self.__name__ = name
        self.__self = arrself

    def __getattr__(self, attr):
        # This only works for the first level of attributes :(
        numpy_symbol = getattr(np, attr)

        if isinstance(numpy_symbol, np.ufunc):
            return _FakeUfunc(numpy_symbol, self.__self)

        elif hasattr(numpy_symbol, "_implementation"):
            # Crude method to find __array_function__ dispatched funcs
            return _BoundFunction(numpy_symbol, self.__self)

        raise AttributeError(f"unsupported numpy function for {self.__name__}")

    def array(self, *args, **kwargs):
        arr = np.array(*args, **kwargs)
        return self.__arraytype_(arr)

    def asarray(self, *args, **kwargs):
        arr = np.asarray(*args, **kwargs)
        return self.__arraytype_(arr)


def inject_array_module(arrtype, known_types, module=None):
    """Inject an __array_module__ function into a type (not instance).

    Parameters
    ----------
    arrtype : class
        A class which to add an array module method to.
    known_types : set or sequence
        types (other than arrtype) which our functions must understand.
    module : module or None
        If a module, this is returned by ``get_array_module``, otherwise
        a fake module is returned which attempts to make use of
        ``__array_ufunc__`` and `__array_function__`` in a very hacky way.

        It is unclear if this can effectively ever work well... And it is
        not really tested. It assumes that ``arrtype(numpy_array)`` is valid.

    Notes
    -----
    This function may not work for types defined in C.
    """
    if module is None:
        if not hasattr(arrtype, "__array_ufunc__"):
            raise ValueError("Array type must support __array_ufunc__")
        if not hasattr(arrtype, "__array_function__"):
            raise ValueError("Array type must support __array_function__")

    def __array_module__(self, types):
        for type in types:
            if type is arrtype:
                continue
            if type in known_types:
                continue
            break
        else:
            if module is not None:
                return module
            return _ArrayFunctionFallback(arrtype.__module__, self)

        return NotImplemented

    arrtype.__array_module__ = __array_module__

