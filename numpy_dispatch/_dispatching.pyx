# cython: language_level=3, bounds_check=False

import warnings

import numpy as _numpy
from numpy cimport ndarray

from cpython cimport (
    PyObject, PyTuple_New, Py_INCREF, PyTuple_SET_ITEM, PyTuple_GET_ITEM)

cdef extern from "Python.h":
    cdef PyObject *PyThreadState_GetDict()
    cdef PyObject *PyEval_GetBuiltins()


__all__ = [
    "get_array_module", "enable_dispatching_globally", "ensure_dispatching",
    "future_dispatch_behavior"]

cdef object numpy = _numpy  # avoid global python module lookup
cdef int _dispatching_globally_enabled = False

cdef Py_ssize_t _dispatching_contexts_used = 0
cdef Py_ssize_t _warning_contexts_used = 0


def enable_dispatching_globally():
    """Globally enable dispatching for `get_array_module`.

    Notes
    -----
    This function should be called *exactly* once by an end-user. It will
    give a warning when called more than once.
    Library authors can use the `ensure_dispatching` context manager to
    achieve the same things locally.
    """
    global _dispatching_globally_enabled

    if _dispatching_globally_enabled:
        warnings.warn(
            "Called `enable_dispatching_globally()` more than once. This "
            "function is meant to be called exactly once by an end-user.",
            UserWarning, stacklevel=1)

    _dispatching_globally_enabled = True


cdef int _is_dispatching_enabled():
    if _dispatching_globally_enabled:
         # The user has globally opted in.
        return True

    if _dispatching_contexts_used == 0:
        return False

    cdef dict local_dict = _get_threadlocal_dict()
    return local_dict.get("__numpy_dispatch_locally_ensured__", False)


cdef dict _get_threadlocal_dict():
    """Fetches the thread local storage dict (taken from np.errstate)
    """
    cdef PyObject *thedict = PyThreadState_GetDict();
    if (thedict == NULL):
        thedict = PyEval_GetBuiltins();

    return <dict>thedict


cdef class ensure_dispatching:
    """Context manager to ensure that dispatching is enabled locally.
    """
    cdef int entered
    def __cinit__(self):
        self.entered = False

    def __enter__(self):
        global _dispatching_contexts_used

        if _is_dispatching_enabled():
            # It is already enabled, so nothing to do.
            # TODO: Could optimize a PyThreadState_GetDict call away here...
            return

        cdef dict local_dict = _get_threadlocal_dict()
        local_dict["__numpy_dispatch_locally_ensured__"] = True
        _dispatching_contexts_used += 1
        self.entered = True

    def __exit__(self, type, value, traceback):
        global _dispatching_contexts_used
        if not self.entered:
            return

        cdef dict local_dict = _get_threadlocal_dict()
        local_dict["__numpy_dispatch_locally_ensured__"] = False
        _dispatching_contexts_used -= 1


cdef int _are_transition_warnings_enabled():
    if _warning_contexts_used == 0:
        return True

    cdef dict local_dict = _get_threadlocal_dict()
    return not local_dict.get("__numpy_dispatch_no_warnings__", False)


cdef class future_dispatch_behavior:
    """Context manager which to opt-in into future behaviour, which is
    either turning a DeprecationWarning into an error or

    Notes
    -----
    Please only use this context manager locally. If this context manager
    is used, in principle an internal function is silently also opted in.
    Basically, we assume that any function which enables dispatching either:

    1. Knows that all of its callees correctly support dispatching
    2. Better, ensure that all of its callees only get input of a well
       defined type for which opting in cannot possibly make a difference.
    """
    cdef int entered
    def __cinit__(self):
        self.entered = False

    def __enter__(self):
        global _warning_contexts_used

        if not _are_transition_warnings_enabled():
            # It is already enabled, so nothing to do.
            # TODO: Could optimize a PyThreadState_GetDict call away here...
            return

        cdef dict local_dict = _get_threadlocal_dict()
        local_dict["__numpy_dispatch_no_warnings__"] = True
        _warning_contexts_used += 1
        self.entered = True

    def __exit__(self, type, value, traceback):
        global _warning_contexts_used

        if not self.entered:
            return

        cdef dict local_dict = _get_threadlocal_dict()
        local_dict["__numpy_dispatch_no_warnings__"] = False
        _warning_contexts_used -= 1


cdef _give_fallback_warning(array_types, int stacklevel, fallback):
    if fallback == "warn":
        if _are_transition_warnings_enabled():
            warnings.warn(
                "Using default because no array-like was able to handle "
                f"all involved types: {array_types!r}. This will raise an error "
                "in the future\n"
                "You can use:\n"
                "    with numpy_dispatch.future_dispatch_behavior()\n"
                "to turn this into an error right away.",
                DeprecationWarning, stacklevel=stacklevel)
            return
    elif fallback is not False:
        raise RuntimeError("Internal error fallback has to be False or 'warn'.")

    raise TypeError(
        "Unable to find a common array type/module for the input array types: "
        f"{array_types!s}. You will have convert at least one of the inputs "
        "to the desired array-object.")


def get_array_module(*args, default=numpy, modules=None, future_modules=False,
                     fallback=False, int stacklevel=1, enabled=None):
    """
    Return the array module based on the passed in objects (or actually
    their types).

    Parameters
    ----------
    *args : array-like objects
        Array arguments for which a common array module should be found.
    default : module, default numpy
        The default module to use, if not given ``numpy`` is assumed.
    modules : None, string, or set/sequence of strings
        If given and not None and the discovered module does not include this
        name, a TypeError is raised.
    future_modules : False, None, string, or set/sequence of strings
        A second set of modules. If the module is not part of `modules`
        but part of `future_modules` a FutureWarning will be given allowing
        the user to opt-in or opt-out (by casting the arrays).
        None means all modules will be allowed, False means no modules are
        transitioning.
    fallback : False, "warn"
        This affects what happens when either no type can handle all inputs
        or the found module is in `future_modules` only.
        In this case a fallback to the `default` may be used, which should
        otherwise only happen if no arrays were passed in.
        If set to False, an error will be given instead. If "warn" is passed,
        a transition warning is given and `default` will be returned.
        This is a DeprecationWarning if no common module is found.
        If an invalid module is found (which is in `future_modules`) it changes
        the `FutureWarning` message. (modules in the `future_modules` always
        give a warning whether fallback is enabled or not).
        Fallback is meant to be used for the initial transition from NumPy
        only. After which it is expected to be False and thus have stricter
        typing.
    stacklevel : int, default 1
        The warning stacklevel to use, defaults to 1. Most library functions
        should set this to 2 or higher so that the user code line is reported.
    enabled : True or None
        Allows forcing the use of dispatching for libraries which wish to
        use it as part of their default API and that are not transitioning.
        This can be used for newly developed code. Libraries should probably
        be consistent and likely most libraries should not use this.
    """
    # Well, this turned out more like C code than cython code :)

    if enabled is None:
        if not _is_dispatching_enabled():
            return default
    elif enabled is not True:
        raise RuntimeError("enabled must be True or None.")

    cdef Py_ssize_t num_args = len(args)

    # Note that array_objects only includes the first one of a given type
    cdef list array_objects = []
    cdef list all_types = []

    cdef Py_ssize_t num_all_types = 0
    cdef Py_ssize_t num_arrays = 0

    cdef Py_ssize_t i, j

    for i in range(num_args):
        pyobj = args[i]
        pytype = type(pyobj)

        for j in range(num_all_types-1, -1, -1):
            if pytype is all_types[j]:
                break
        else:
            all_types.append(pytype)
            num_all_types += 1

            if pytype is ndarray or hasattr(pytype, "__array_module__"):
                # Either append the array, or insert it if it is a subclass
                # of a previous one, this makes sure subclasses are inserted
                # before classes.
                # TODO: This implementation sorts the type tuple, that is
                #       probably OK, but not specified (and not in the NEP)
                for j in range(num_arrays):
                    if issubclass(type(pyobj), type(array_objects[j])):
                        array_objects.insert(j, pyobj)
                        break
                else:
                    array_objects.append(pyobj)
                num_arrays += 1

    best_type = None

    if num_arrays == 0:
        # We return the default if no array types are found.
        return default

    if num_arrays == 1 and type(array_objects[0]) is ndarray:
        # If NumPy is the only type, NumPy will be chosen, otherwise
        # NumPy will always defer (subclasses are not handled here)
        module, best_type = numpy, ndarray

    else:
        # Note: In theory we could add a super fast cache for this tuple?
        pytype_tuple = PyTuple_New(num_arrays)
        for i in range(num_arrays):
            PyTuple_SET_ITEM(pytype_tuple, i, type(array_objects[i]))

        for i in range(num_arrays):
            arr = array_objects[i]
            array_type = type(arr)
            if array_type is ndarray:
                # Ignore NumPy (handled earlier)
                continue

            # call the actual method, we look up __array_module__ twice,
            # that could be optimized away.
            res = array_type.__array_module__(arr, pytype_tuple)
            if res is NotImplemented:
                continue
            module = res
            best_type = array_type
            break
        else:
            # All types returned NotImplemented, if fallback is "warn"
            # use the default, but give a fallback warning unless the user
            # turns it into an error. If fallback is False, always raise
            # the error.
            _give_fallback_warning(pytype_tuple, stacklevel, fallback)

            return default

    cdef str module_name = module.__name__

    # If module_name is included in `modules` we are finished:
    if modules is None:
        return module
    elif type(modules) is str and modules == module_name:
        return module
    elif module_name in modules:
        return module

    # Otherwise it may still be OK based on a transition.
    cdef int future_support
    if future_modules is False:
        future_support = False
    elif future_modules is None:
        future_support = True
    elif type(future_modules) is str and future_modules == module_name:
        future_support = True
    elif module_name in future_modules:
        future_support = True
    else:
        future_support = False

    if not future_support:
        # There will not be future support (same error can occur when fallback
        # is false.
        raise TypeError(
            f"Array module {module_name} for array-type {best_type!s} "
            f"found, but only {modules!r} are supported.")

    # The user gave type, that will be supported in the future.
    # If the warnings are disabled, we opt-in to the new behaviour
    # (doing nothing). Otherwise, we warn and return the (old) default.
    if not _are_transition_warnings_enabled():
        return module

    if fallback is False:
        warnings.warn(
            "In the future this function will use the array-module "
            f"{module_name} of the array=type {best_type!s} instead of "
            "raising an error.\n"
            "To opt-out of this behaviour, cast your array manually. "
            "To opt-in to the new behaviour use:\n"
            "    with numpy_dispatch.future_dispatch_behavior():",
            FutureWarning, stacklevel=stacklevel)

        raise TypeError(
            f"Array module {module_name} for array-type {best_type!s} "
            f"found, but only {modules!r} are supported.")

    elif fallback == "warn":
        warnings.warn(
            "In the future this function will stop interpreting "
            f"these input arrays using the module {default.__name__}. "
            f"Instead it will use the discovered module {module_name}.\n"
            "To opt-out of this behaviour, cast your array manually. "
            "To opt-in to the new behaviour use:\n"
            "    with numpy_dispatch.future_dispatch_behavior():",
            FutureWarning, stacklevel=stacklevel)

        return default
    else:
        raise RuntimeError("Internal error fallback has to be False or 'warn'.")


