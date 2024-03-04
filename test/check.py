# ==========================================================================
#
#   Copyright NumFOCUS
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#          https://www.apache.org/licenses/LICENSE-2.0.txt
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ==========================================================================

from __future__ import annotations

from typing import Any, Sequence, Union

import numpy as np
from numpy.typing import NDArray

NDArrayFloat = NDArray[np.float_]
NDArrayInt = NDArray[np.int_]
SequenceFloat = Union[Sequence[float], NDArrayFloat]
SequenceInt = Union[Sequence[int], NDArrayInt]


def check_deeply_numeric(x: Any) -> bool:
    return isinstance(x, (int, float, np.float32, np.float64)) or (
        isinstance(x, np.ndarray)
        and (
            (len(x.shape) == 0 and check_deeply_numeric(x[()]))
            or (len(x.shape) > 0 and all(check_deeply_numeric(e) for e in x))
        )
    )


def deeply_allclose(a: Any, b: Any, rtol=1e-05, atol=1e-08, equal_nan=False) -> bool:
    response: bool
    response = (
        isinstance(b, (int, float, np.int32, np.int64, np.float32, np.float64))
        and np.allclose(a, b, rtol, atol, equal_nan)
        if (isinstance(a, (int, float, np.int32, np.int64, np.float32, np.float64)))
        else False
        if type(a) is not type(b)
        else deeply_allclose(a.shape, b.shape, rtol, atol, equal_nan)
        and np.allclose(a, b, rtol, atol, equal_nan)
        if isinstance(a, np.ndarray)
        else a == b
        if isinstance(a, str)
        else len(a) == len(b)
        and all(
            deeply_allclose(elem_a, elem_b, rtol, atol, equal_nan)
            for elem_a, elem_b in zip(a, b)
        )
        if isinstance(a, (list, tuple, set))
        else deeply_allclose(set(a.keys()), set(b.keys()), rtol, atol, equal_nan)
        and all(deeply_allclose(a[k], b[k], rtol, atol, equal_nan) for k in a.keys())
        if isinstance(a, dict)
        else False
    )
    if False and not response:
        print("response = False")
        print(f"{type(a) = }")
        print(f"{type(b) = }")
        print(f"{a = }")
        print(f"{b = }")
    return response


def deep_print(x) -> str:
    return (
        f"{x:.20g}"
        if isinstance(x, (float, np.float32, np.float64))
        else repr(x)
        if isinstance(x, (int, np.int32, np.int64, str))
        else deep_print_array(x)
        if isinstance(x, np.ndarray)
        else deep_print_tuple(x)
        if isinstance(x, tuple)
        else deep_print_list(x)
        if isinstance(x, list)
        else deep_print_dict(x)
        if isinstance(x, dict)
        else deep_print_set(x)
        if isinstance(x, set)
        else repr(type(x))
    )


def deep_print_array(x) -> str:
    return (
        "np.array("
        + (deep_print_array_iterable(x) if len(x.shape) else deep_print(x[()]))
        + ")"
    )


def deep_print_array_iterable(x) -> str:
    return (
        "["
        + (
            ", ".join(deep_print_array_iterable(e) for e in x)
            if len(x.shape) > 1
            else ", ".join(deep_print(e) for e in x)
        )
        + ",]"
    )


def deep_print_list(x) -> str:
    return "[" + ", ".join(deep_print(e) for e in x) + ",]" if len(x) else repr(list())


def deep_print_tuple(x) -> str:
    return "(" + ", ".join(deep_print(e) for e in x) + ",)" if len(x) else repr(tuple())


def deep_print_set(x) -> str:
    return "{" + ", ".join(deep_print(e) for e in x) + ",}" if len(x) else repr(set())


def deep_print_dict(x) -> str:
    return (
        "{"
        + ", ".join(deep_print(k) + ": " + deep_print(v) for k, v in x.items())
        + ",}"
        if len(x)
        else repr(dict())
    )
