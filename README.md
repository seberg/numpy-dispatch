# Get Array Module

This package is an implementation for NumPy's `get_array_module` proposal.
This proposal is written up in [NEP 37](https://numpy.org/neps/nep-0037-array-module.html).

## For End-Users

End users will not use much of the module, except to enable the behaviour
to begin with or get control over `FutureWarnings`.
In some cases a library is also a (localized) end-user.

For end users, there are two functionalities available:
```python
import numpy_dispatch

numpy_dispatch.enable_dispatching_globally()
```
To globally enable dispatching. This function is meant to be called
*exactly* once and *only* by the end-user.
Calling the function more than once currently gives a warning.
The global switch *cannot* be disabled.

In some cases local control may be necessary. Locally enabling dispatching
can be achieved in a thread-safe manner by using:
```python
import numpy_dispatch

with numpy_dispatch.ensure_dispatching():
    dispatching_aware_library_function(arr1, arr2)
```
We currently assume that disabling dispatching is never necessary.
To effectively disable dispatching, it will be necessary to cast all inputs
to NumPy arrays.


## For library authors

Library authors are the main audience for this, the central function
is `get_array_module`. A typical use case may look like this:

```python
import numpy as np

try:
    from numpy_dispatch import get_array_module
except
    # Simply use NumPy if unavailable
    get_array_module = lambda *args, **kwargs: np


def library_function(arr1, arr2, **other_parameters):
    onp = get_array_module(arr1, arr2)

    arr1 = onp.asarray(arr1)
    arr2 = onp.asarray(arr2)

    # use onp instead of all numpy code
```

If your library has internal helper functions, the best way to write these
is probably:
```python
def internal_helper(*args, onp=np):
   # old code, but replace `np` with `onp`.
```
That way the module is passed around in a safe manner. If your library calls
into other libraries which may or may not dispatch (or even dispatch in the
future), it is probably best to ensure that all types tidy so that dispatching
is not expected to make a difference if enabled.
Remember, you do _not_ have control to _disable_ dispatching.


### More finegrained gontrol

To enable transition towards allowing certain or all types, there are a few
additional options, for example:
```python
onp = get_array_module(*arrays, modules="numpy", future_modules="dask.array")
```
could be a spelling to say that currently NumPy is fully supported, and `dask`
is supported, but the support will warn (the user can silence the warning).

Giving `future_modules=None` would allow any and all module to be returned.

Library authors further have the option to provide the `default`, in case the
library is originally written for something other than NumPy.
Additionally, since some libraries may want to always use dispatching as
a design choice, `enabled={True, None}` can currently be passed in.
Disabling dispatching is currently specifically not supported.


## Transitioning for users and libraries

The `future_modules` keyword argument is a way to transition users (enabling
new dispatching).
When such a transition happens two things change:

1. When no array-module can be found because none of the array types
   understands the others, a `default` has to be returned during
   a transition phase, this can be done using `fallback="warn"`
2. `future_modules` allows to implement new modules, but not use them
   by default (give a `FutureWarning` instead).

In both cases, to opt-out of the change, the user will have to cast their
input arrays.
To opt-in to the new changes (error in 1., dispatching in 2.), the user
can locally use the context manager:
```python
from numpy_dispatch import future_dispatch_behavior

with future_dispatch_behavior():
    library_function_doing_a_transition()
```


## Array object implementors

Please read [NEP 37](https://numpy.org/neps/nep-0037-array-module.html).
The only addition here is that `module.__name__` has to return a good
and stable name to be used by libraries.


# Testing!

To allow some (silly) testing there are two functionalities to make it
a bit simpler:

```python
from numpy_dispatch import dummy  # will give a warning

# Create a dummy array, that returns its own module, although that
# module effectively is just NumPy. The dummy array accepts only NumPy
# arrays aside from itself.
dummy.array([1, 2, 3])

# Add `__array_module__` which accepts the given types as types that
# it can understand. This example is with cupy (untested), which is
# probably the best module to test:
import cupy
dummy.inject_array_module(cupy.ndarray, (np.ndarray, dummy.DummyArray,),
                          module=cupy)
```
