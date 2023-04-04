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
import scipy.stats
from numpy.typing import NDArray
from typing import Mapping, MutableMapping, Sequence, Any


class ComputeCertainty:
    all_certainty_types = ["negative_entropy", "confidence", "margin"]

    def __init__(self, certainty_type, percentiles, cutoffs):
        """
        certainty_type can be "negative_entropy", "confidence", or "margin", or a list
        (or tuple) of one or more of these.  A value of None means all certainty types.

        percentiles is a list (or tuple) of percentile values to be computed, e.g., (5,
        10, 25, 50).  The P percentile score will be that score S such that P percent of
        all scores are less than (more uncertain than) S.

        cutoffs: if the `certainty_type` is a single type then `cutoffs` can be a list
        (or tuple) of computed scores to compute the cumulative distribution function
        value for.  For example for "confidence" it could be (0.5, 0.75, 0.9, 0.95).
        The cdf value for S will the percentage P of all scores that are less than (more
        uncertain than) S.  If `certainty_type` includes multiple types (including the
        case that `certainty_type` is None) then `cutoffs` is a dictionary whose keys
        are the certainty types and each value is the list (tuple) of cutoffs for that
        certainty type.
        """

        if certainty_type is None:
            certainty_type: Sequence[str] = self.all_certainty_types
        if isinstance(certainty_type, str):
            certainty_type = [certainty_type]
        if not all([ct in self.all_certainty_types for ct in certainty_type]):
            raise ValueError("Something wrong with certainty_type")
        self.certainty_type: Sequence[str] = certainty_type

        if percentiles is None:
            # No percentiles
            percentiles = []
        percentiles = [float(p) for p in percentiles]
        self.percentiles: Sequence[float] = percentiles

        if cutoffs is None:
            # No cutoffs for any certainty type
            cutoffs = {ct: [] for ct in certainty_type}
        if len(certainty_type) == 1 and not isinstance(cutoffs, dict):
            cutoffs = {certainty_type[0]: cutoffs}
        cutoffs = {key: [float(c) for c in value] for key, value in cutoffs.items()}
        if not set(cutoffs.keys()) == set(certainty_type):
            raise ValueError("Something wrong with cutoffs")
        self.cutoffs: Mapping[str, Sequence[float]] = cutoffs

    def from_numpy_array(self, predictions: NDArray) -> Mapping[str, Mapping[str, Any]]:
        # Compute several certainty scores for each prediction.  High scores correspond
        # to high certainty.  Also report summary statistcs about these scores.

        # The `predictions` variable may have 2-dimensional shape with indexes (example,
        # class) or 3-dimensional shape with indexes (example, random_sample, class).
        # This code respects this variety in dimensions.  Regardless, it is assumed that
        # the last dimension is over the classes (a.k.a. the possible values for the
        # label.)

        num_predictions: int = predictions.size // predictions.shape[-1]

        if np.amax(predictions) <= 0.0:
            # If no values are positive then these are log_softmax values.  With
            # logarithms, scaling the corresponding non-logarithmic values
            # proportionately means adding the same constant to each logarithm.  We
            # scale the softmax values in a row so that the maximum value won't overflow
            # or underflow when we exponentiate.  The scaling is otherwise immaterial
            # because we later normalize softmax values for a row to sum to 1.0.
            predictions = predictions - np.amax(predictions, axis=-1, keepdims=True)
            # Convert log_softmax to softmax
            predictions = np.exp(predictions)

        # Normalize rows to sum to 1.0
        predictions = predictions / np.sum(predictions, axis=-1, keepdims=True)
        # Find the sorted order of values within each row.
        predictions = np.sort(predictions, axis=-1)

        scores: MutableMapping[str, NDArray] = dict()
        # When certainty is defined by entropy, compute the entropy of each row and
        # negate it, because we want larger values (i.e. values that are less negative)
        # to represent more certainty than smaller values.
        if "negative_entropy" in self.certainty_type:
            scores["negative_entropy"] = -scipy.stats.entropy(predictions, axis=-1)
        # When certainty is defined by confidence, use the largest prediction
        # probability.
        if "confidence" in self.certainty_type:
            scores["confidence"] = predictions[..., -1]
        # When certainty is defined by margin, use the difference between the largest
        # and second largest prediction probabilities.
        if "margin" in self.certainty_type:
            scores["margin"] = predictions[..., -1] - predictions[..., -2]

        # Report percentile scores
        percentile: float
        percentile_score: float
        cutoff: float
        source_name: str

        response: Mapping[str, Mapping[str, Any]] = {
            source_name: {
                "scores": scores[source_name],
                "percentiles": {
                    percentile: percentile_score
                    for percentile, percentile_score in zip(
                        self.percentiles,
                        np.percentile(scores[source_name], self.percentiles),
                    )
                },
                "cdf": {
                    cutoff: (scores[source_name] < cutoff).sum() / num_predictions * 100
                    for cutoff in self.cutoffs[source_name]
                },
            }
            for source_name in scores.keys()
        }

        return response