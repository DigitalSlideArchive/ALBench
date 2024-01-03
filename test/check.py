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
import numpy as np
from typing import Any


def check_deeply_numeric(x: Any) -> bool:
    return isinstance(x, (int, float, np.float32, np.float64)) or (
        isinstance(x, np.ndarray)
        and (
            (len(x.shape) == 0 and check_deeply_numeric(x[()]))
            or (len(x.shape) > 0 and all([check_deeply_numeric(e) for e in x]))
        )
    )


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
