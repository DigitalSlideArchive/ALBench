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
import al_bench as alb
import al_bench.factory
import numpy as np
import random
import torch
from data_0050_factory import expected_certainties
from check import deep_print
from numpy.typing import NDArray
from typing import Any, Dict, List, Mapping, Type


def build_predictions(num_samples: int, num_repeats: int, num_classes: int) -> NDArray:
    predictions: NDArray
    # Make predictions array where each row sums to 1.0.  (In fancy words, we are using
    # a Dirichlet distribution, where each class has a single pseudocount.)  It is
    # overkill, but we will generate random 52-bit ints rather than random
    # double-precision floats because the latter give more precision beyond the decimal
    # point for values less than 0.5.
    remove_repeats = num_repeats is None
    if remove_repeats:
        num_repeats = 1

    torch.manual_seed(20240102)
    rng = np.random.default_rng(seed=20240102)
    max_precision: int = 2**52
    predictions = (
        rng.integers(0, max_precision, (num_samples, num_repeats, num_classes - 1))
        / max_precision
    )
    predictions = np.sort(predictions, axis=-1)
    predictions = np.diff(predictions, axis=-1, prepend=0, append=1.0)

    # If the user requested a two-dimensional array then return that
    if remove_repeats:
        predictions = np.reshape(predictions, (num_samples, num_classes))
    return predictions


def test_0050_factory() -> None:
    num_samples = 100
    num_repeats = 20
    num_classes = 5
    predictions = build_predictions(num_samples, num_repeats, num_classes)

    percentiles: NDArray = np.fromiter(range(11), dtype=float) * 10
    if False:
        # Read the values that we expect the percentile calculations to return.  These
        # can be submitted as cutoffs and should return close the original percentiles
        # that generated them.
        cutoffs = {
            k: [cut for cut in v["percentiles"].values()]
            for k, v in expected_certainties.items()
        }
        print(f"cutoffs = {deep_print(cutoffs)}")
        return
    cutoffs = {
        "confidence": [
            0.23319522573892670358,
            0.32432748123291815778,
            0.35435025360693872543,
            0.38084620724981499418,
            0.40770838473435910831,
            0.43709436017449876388,
            0.46737826053936193382,
            0.50387446133793900582,
            0.55052071336584962324,
            0.62265907605418024939,
            0.92456310666481122951,
        ],
        "margin": [
            0.00037661163756985693851,
            0.028640871329089056391,
            0.054366109196087329669,
            0.083985899187714083136,
            0.11950722442761438358,
            0.15983776715969899573,
            0.20349967771826005247,
            0.2592390540508386354,
            0.32837790495088298171,
            0.43819030712335260702,
            0.89597557461394683465,
        ],
        "negative_entropy": [
            -1.6018099636517608708,
            -1.4893020660420734913,
            -1.4373000706607712562,
            -1.3945333933939410009,
            -1.3550938452258365352,
            -1.3144531931864329444,
            -1.2699945981319848975,
            -1.2158309850245696548,
            -1.1445573754943771938,
            -1.0369674780469477771,
            -0.36344068987929223358,
        ],
        "batchbald": [
            0.42628003436274486404,
            2.6369226698467573833,
            2.9110099644887497128,
            2.9374140708677480838,
            2.9585105399558986328,
            2.9742478406972487903,
            2.983767717379953055,
            2.9979672343857641792,
            3.0192880442387632911,
            3.0535196123103358623,
            3.1752039746139502085,
        ],
    }

    compute_certainty: alb.factory.ComputeCertainty
    compute_certainty = alb.factory.ComputeCertainty(
        certainty_type=alb.factory.ComputeCertainty.all_certainty_types,
        percentiles=percentiles,
        cutoffs=cutoffs,
    )
    certainties: Mapping[str, Mapping[str, Any]] = compute_certainty.from_numpy_array(
        predictions
    )

    if True:
        # Until we convince torch to be deterministic or find some other solution,
        # disable checking the results of BatchBALD.
        del certainties["batchbald"], expected_certainties["batchbald"]
        passed = deep_print(certainties) == deep_print(expected_certainties)
        print(f"test_0050_factory() {'passed' if passed else 'failed'}")
        if not passed:
            raise ValueError("test_0050_factory() failed")
    else:
        # Generate expected output for data_0050_factory.py
        print("import numpy as np")
        print(f"expected_certainties = {deep_print(certainties)}")


if __name__ == "__main__":
    test_0050_factory()
