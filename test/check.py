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


def check_deeply_numeric(x: Any):
    return (
        isinstance(x, (int, float, np.float32))
        or (
            isinstance(x, np.ndarray)
            and len(x.shape) == 0
            and check_deeply_numeric(x[()])
        )
        or all([check_deeply_numeric(e) for e in x])
    )
