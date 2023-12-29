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


def deep_array(x) -> str:
    if len(x.shape) == 0:
        return deep_print(x[()])
    if len(x.shape) == 1:
        return "[" + ", ".join(deep_print(e) for e in x) + ",]"
    if len(x.shape) > 1:
        return "[" + ", ".join(deep_array(e) for e in x) + ",]"


def deep_print(x) -> str:
    if isinstance(x, list):
        if len(x):
            return "[" + ", ".join(deep_print(e) for e in x) + ",]"
        else:
            return repr(list())
    if isinstance(x, tuple):
        if len(x):
            return "(" + ", ".join(deep_print(e) for e in x) + ",)"
        else:
            return repr(tuple())
    if isinstance(x, set):
        if len(x):
            return "{" + ", ".join(deep_print(e) for e in x) + ",}"
        else:
            return repr(set())
    if isinstance(x, dict):
        if len(x):
            return (
                "{"
                + ", ".join(deep_print(k) + ": " + deep_print(v) for k, v in x.items())
                + ",}"
            )
        else:
            return repr(dict())
    if isinstance(x, np.ndarray):
        return "np.array(" + deep_array(x) + ")"
    if isinstance(x, (int, np.int32, np.int64, str)):
        return repr(x)
    if isinstance(x, (float, np.float32, np.float64)):
        return f"{x:.20g}"
    return repr(type(x))
