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

from typing import Sequence

import numpy as np

import al_bench as alb
import al_bench.factory
from check import NDArrayFloat, deep_print


def test_0040_factory() -> None:
    # Try one of these certainty_type values
    # certainty_type = "confidence"
    # certainty_type = ("negative_entropy",)
    # certainty_type = ["margin"]
    certainty_type = ["negative_entropy", "confidence", "margin"]
    # certainty_type = None  # None means compute all types

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
    # Create dummy input for testing
    predictions: NDArrayFloat
    predictions = np.array(
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

    if False:
        print(f"expected_output = {deep_print(output)}")

    # Check the output
    # print(f"expected_output = {repr(output)}")
    expected_output = {
        "negative_entropy": {
            "scores": np.array(
                [
                    [
                        -1.4921154215367835594,
                        -1.5649893749522698627,
                        -1.1758272339516300242,
                        -1.3416198259801304093,
                    ],
                    [
                        -0.053655542695062230507,
                        -1.4476777168415093655,
                        -1.089588058934530812,
                        -1.0554713545223466387,
                    ],
                    [
                        -1.4833408253717623726,
                        -1.3288283111504388678,
                        -0.42248202857772260144,
                        -0.47654821488531184936,
                    ],
                ]
            ),
            "percentiles": {
                5: -1.5249087005737524958,
                10: -1.4912379619202813519,
                25: -1.4565934939740725618,
                50: -1.252327772551034446,
            },
            "cdf": {
                -1.5: 8.3333333333333321491,
                -1: 75,
                -0.5: 75,
                -0.2999999999999999889: 91.666666666666657193,
                -0.2000000000000000111: 91.666666666666657193,
                -0.10000000000000000555: 91.666666666666657193,
            },
        },
        "confidence": {
            "scores": np.array(
                [
                    [
                        0.3801900779408526887,
                        0.27613462626882867568,
                        0.42694469852897437567,
                        0.36615065736529744944,
                    ],
                    [
                        0.99220383425516078812,
                        0.35788515375906571059,
                        0.5506100365389926532,
                        0.51319874672684695494,
                    ],
                    [
                        0.41112333251754523689,
                        0.4317528566153233105,
                        0.91000817654746524177,
                        0.84755175430788776136,
                    ],
                ]
            ),
            "percentiles": {
                5: 0.32109741638845906708,
                10: 0.35871170411968889002,
                25: 0.37668022279696389276,
                50: 0.42934877757214884308,
            },
            "cdf": {
                0.2999999999999999889: 8.3333333333333321491,
                0.4000000000000000222: 33.333333333333328596,
                0.5: 58.333333333333335702,
                0.75: 75,
                0.9000000000000000222: 83.333333333333342807,
                0.94999999999999995559: 91.666666666666657193,
            },
        },
        "margin": {
            "scores": np.array(
                [
                    [
                        0.17509739520592343398,
                        0.023253613200393818961,
                        0.083901775776742237856,
                        0.046701556226295126706,
                    ],
                    [
                        0.98858636790614584644,
                        0.078949041812879094948,
                        0.26034974764190493834,
                        0.13776808873166995761,
                    ],
                    [
                        0.22259658142286140037,
                        0.10036973397957560383,
                        0.87967837291878114847,
                        0.70585649091257152143,
                    ],
                ]
            ),
            "percentiles": {
                5: 0.03614998186463953822,
                10: 0.04992630478495352353,
                25: 0.082663592285776452129,
                50: 0.1564327419687966958,
            },
            "cdf": {
                0.050000000000000002776: 16.666666666666664298,
                0.10000000000000000555: 33.333333333333328596,
                0.2999999999999999889: 75,
                0.5: 75,
                0.69999999999999995559: 75,
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
        output[key0][key1] == expected_output[key0][key1]
        for key0 in expected_output_keys | output.keys()
        for key1 in expected_output[key0].keys() | output[key0].keys()
        if key1 != "scores"
    )
    assert all(
        ((output[key]["scores"] - expected_output[key]["scores"]) ** 2).sum() < 1e-12
        for key in expected_output_keys | output.keys()
    )


if __name__ == "__main__":
    test_0040_factory()
