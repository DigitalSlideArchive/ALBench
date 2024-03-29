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

from typing import Any, Mapping, Type

import al_bench as alb
import al_bench.model
from exercise import exercise_model_handler


def test_0020_model_handler_interface() -> None:
    """Purpose: Test that high-level operations work"""

    # Specify some testing parameters
    parameters: Mapping[str, Any]
    parameters = {
        "number_of_superpixels": 1000,
        "number_of_features": 2048,
        "number_of_categories_by_label": [5, 7],
        "label_to_test": 0,
    }

    # Try trivial exercises on the handler interface
    ModelHandler: Type[alb.model.AbstractModelHandler]
    for ModelHandler in (
        alb.model.TensorFlowModelHandler,
        alb.model.PyTorchModelHandler,
    ):
        my_model_handler: alb.model.AbstractModelHandler = ModelHandler()
        exercise_model_handler(my_model_handler, **parameters)


if __name__ == "__main__":
    test_0020_model_handler_interface()
