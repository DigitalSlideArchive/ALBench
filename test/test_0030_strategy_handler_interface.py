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

import al_bench as alb
from exercise import exercise_strategy_handler


def test_0030_strategy_handler_interfac():
    """Purpose: Test that high-level operations work"""

    # Specify some testing parameters
    parameters = {
        "number_of_superpixels": 1000,
        "number_of_features": 2048,
        "number_of_categories_by_label": [5, 7],
        "label_to_test": 0,
    }

    # Try trivial exercises on the handler interface
    for StrategyHandler in (
        alb.strategy.RandomStrategyHandler,
        alb.strategy.LeastConfidenceStrategyHandler,
        alb.strategy.LeastMarginStrategyHandler,
        alb.strategy.EntropyStrategyHandler,
    ):
        my_strategy_handler = StrategyHandler()
        exercise_strategy_handler(my_strategy_handler, **parameters)


if __name__ == "__main__":
    test_0030_strategy_handler_interfac()
