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
import al_bench as alb
import al_bench.factory
from numpy.typing import NDArray
from typing import Sequence


def test_0040_factory() -> None:
    # Try one of these certainty_type values
    # certainty_type = "confidence"
    # certainty_type = ("negative_entropy",)
    # certainty_type = ["margin"]
    # certainty_type = ["confidence", "margin"]
    certainty_type = None  # None means compute all types

    # Try one of these lists of percentiles
    # percentiles = None  # None means don't compute these
    percentiles: Sequence[float] = (5, 10, 25, 50)

    # Try one of these lists of cutoffs
    # cutoffs = None  # None means don't compute these
    cutoffs = {
        "confidence": (0.3, 0.4, 0.5, 0.75, 0.9, 0.95),
        "margin": (0.05, 0.1, 0.3, 0.5, 0.7),
        "negative_entropy": (-1.5, -1.0, -0.5, -0.3, -0.2, -0.1),
    }
    # Depending on which certainty_type we choose above, we may have supplied too much
    # cutoff information; keep only what is called for.
    if cutoffs is not None:
        cutoffs = {
            key: value
            for key, value in cutoffs.items()
            if certainty_type is None
            or (isinstance(certainty_type, str) and key == certainty_type)
            or key in certainty_type
        }

    # Get a class from our factory
    certainty_computer = alb.factory.ComputeCertainty(
        certainty_type, percentiles, cutoffs
    )
    print(f"all_certainty_types = {certainty_computer.all_certainty_types}")
    # Create dummy input for testing
    predictions: NDArray = np.array(
        [
            [
                [6.32163380, 3.41019114, 3.30049889, 2.06819065, 1.52704675],
                [2.06655387, 2.58424900, 2.82188300, 1.47125306, 1.27529014],
                [10.5860758, 0.215300724, 1.57858528, 22.9794548, 18.4636075],
                [50.4311630, 27.9772560, 13.4680106, 43.9988004, 1.85813736],
            ],
            [
                [10.2926716, 11.1291663, 1.18709917, 3052.52362, 1.37603382],
                [3.01946865, 9.66897743, 1.47935312, 5.31317224, 7.53601244],
                [1.23057158, 52.1269583, 11.6077417, 2.22669709, 27.4793138],
                [2.29362951, 7.21740315, 38.2531577, 52.2905420, 1.83667391],
            ],
            [
                [1.61233572, 2.72736889, 5.94761741, 2.15977544, 2.01964996],
                [4.98827645, 1.03082409, 1.31308406, 6.49913185, 1.22158187],
                [5.34407778, 4.90344745, 160.342432, 3.44236429, 2.16657002],
                [2.12329553, 4.45499601, 1.57992621, 107.503275, 643.032005],
            ],
        ]
    )

    # Process the inputs with our factory class
    output = certainty_computer.from_numpy_array(predictions)

    # Check the output
    # print(f"expected_output = {repr(output)}")
    expected_output = {
        "negative_entropy": {
            "scores": np.array(
                [
                    [-1.49211542, -1.56498937, -1.17582723, -1.34161983],
                    [-0.05365554, -1.44767772, -1.08958806, -1.05547135],
                    [-1.48334083, -1.32882831, -0.42248203, -0.47654821],
                ]
            ),
            "percentiles": {
                5.0: -1.5249087005737525,
                10.0: -1.4912379619202816,
                25.0: -1.4565934939740726,
                50.0: -1.2523277725510344,
            },
            "cdf": {
                -1.5: 8.333333333333332,
                -1.0: 75.0,
                -0.5: 75.0,
                -0.3: 91.66666666666666,
                -0.2: 91.66666666666666,
                -0.1: 91.66666666666666,
            },
        },
        "confidence": {
            "scores": np.array(
                [
                    [0.38019008, 0.27613463, 0.4269447, 0.36615066],
                    [0.99220383, 0.35788515, 0.55061004, 0.51319875],
                    [0.41112333, 0.43175286, 0.91000818, 0.84755175],
                ]
            ),
            "percentiles": {
                5.0: 0.32109741638845907,
                10.0: 0.3587117041196889,
                25.0: 0.3766802227969639,
                50.0: 0.42934877757214884,
            },
            "cdf": {
                0.3: 8.333333333333332,
                0.4: 33.33333333333333,
                0.5: 58.333333333333336,
                0.75: 75.0,
                0.9: 83.33333333333334,
                0.95: 91.66666666666666,
            },
        },
        "margin": {
            "scores": np.array(
                [
                    [0.1750974, 0.02325361, 0.08390178, 0.04670156],
                    [0.98858637, 0.07894904, 0.26034975, 0.13776809],
                    [0.22259658, 0.10036973, 0.87967837, 0.70585649],
                ]
            ),
            "percentiles": {
                5.0: 0.03614998186463954,
                10.0: 0.049926304784953524,
                25.0: 0.08266359228577645,
                50.0: 0.1564327419687967,
            },
            "cdf": {
                0.05: 16.666666666666664,
                0.1: 33.33333333333333,
                0.3: 75.0,
                0.5: 75.0,
                0.7: 75.0,
            },
        },
    }
    # Depending on the values supplied to the factory, we may not get all of our
    # anticipated expected_output.  Check for only what we should have actually gotten.
    expected_output_keys = {
        key
        for key in expected_output.keys()
        if certainty_type is None
        or (isinstance(certainty_type, str) and key == certainty_type)
        or key in certainty_type
    }
    assert all(
        [
            output[key0][key1] == expected_output[key0][key1]
            for key0 in expected_output_keys | output.keys()
            for key1 in expected_output[key0].keys() | output[key0].keys()
            if key1 != "scores"
        ]
    )
    assert all(
        [
            ((output[key]["scores"] - expected_output[key]["scores"]) ** 2).sum()
            < 1e-12
            for key in expected_output_keys | output.keys()
        ]
    )


if __name__ == "__main__":
    test_0040_factory()
