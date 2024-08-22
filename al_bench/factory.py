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

from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import scipy.stats
from numpy.typing import NDArray


class ComputeCertainty:
    confidence: str = "confidence"
    margin: str = "margin"
    negative_entropy: str = "negative_entropy"
    batchbald: str = "batchbald"
    all_certainty_types: List[str]
    all_certainty_types = [confidence, margin, negative_entropy, batchbald]

    def __init__(self, certainty_type, percentiles, cutoffs) -> None:
        """
        certainty_type can be "confidence", "margin", "negative_entropy", "batchbald" or
        a list (or tuple) of one or more of these.  A value of None means all certainty
        types.

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
            certainty_type = self.all_certainty_types
        if isinstance(certainty_type, str):
            certainty_type = [certainty_type]
        if not all(ct in self.all_certainty_types for ct in certainty_type):
            raise ValueError(f"Something wrong with {certainty_type = }")
        self.certainty_type: Sequence[str] = certainty_type

        if percentiles is None:
            # No percentiles
            percentiles = []
        percentiles = [float(p) for p in percentiles]
        self.percentiles: Sequence[float] = percentiles

        if cutoffs is None:
            cutoffs = {}
        if len(certainty_type) == 1 and isinstance(cutoffs, (list, tuple)):
            cutoffs = {certainty_type[0]: cutoffs}
        # If we have no information for a certainty type, default to no cutoffs.
        cutoffs = {**{k: [] for k in certainty_type}, **cutoffs}
        if not all(cut in certainty_type for cut in cutoffs.keys()):
            raise ValueError(f"Something wrong with {cutoffs = }")
        cutoffs = {key: [float(c) for c in value] for key, value in cutoffs.items()}
        self.cutoffs: Mapping[str, Sequence[float]] = cutoffs

        # Defaults.  These can be overridden with setter methods.
        self.batchbald_batch_size: int = 100
        self.batchbald_num_samples: int = 10000
        self.batchbald_excluded_samples: NDArray[np.int_] = np.array([], dtype=np.int_)

    def set_batchbald_batch_size(self, batchbald_batch_size: int) -> None:
        self.batchbald_batch_size = batchbald_batch_size

    def set_batchbald_num_samples(self, batchbald_num_samples: int) -> None:
        self.batchbald_num_samples = batchbald_num_samples

    def set_batchbald_excluded_samples(self, exc_samples: NDArray[np.int_]) -> None:
        self.batchbald_excluded_samples = exc_samples

    def from_numpy_array(
        self, predictions: NDArray[np.float_]
    ) -> Mapping[str, Mapping[str, Any]]:
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
        # Find the two largest values within each row.
        partitioned = np.partition(predictions, -2, axis=-1)[..., -2:]

        scores: Dict[str, NDArray[np.float_]] = dict()
        # When certainty is defined by confidence, use the largest prediction
        # probability.
        if "confidence" in self.certainty_type:
            scores["confidence"] = partitioned[..., -1].copy()

        # When certainty is defined by margin, use the difference between the largest
        # and second largest prediction probabilities.
        if "margin" in self.certainty_type:
            scores["margin"] = partitioned[..., -1] - partitioned[..., -2]

        # When certainty is defined by entropy, compute the entropy of each row and
        # negate it, because we want larger values (i.e. values that are less negative)
        # to represent more certainty than smaller values.
        if "negative_entropy" in self.certainty_type:
            scores["negative_entropy"] = -scipy.stats.entropy(predictions, axis=-1)

        # When certainty is determined by batchbald, we let batchbald_redux do our
        # calculations.
        if "batchbald" in self.certainty_type:
            scores["batchbald"] = self.batchbald_scores(predictions)

        # Report scores, percentile scores, and cutoff percentiles
        response: Mapping[str, Mapping[str, Any]]
        response = {
            source_name: {
                "scores": scores[source_name],
                "percentiles": dict(
                    zip(
                        self.percentiles,
                        np.percentile(scores[source_name], self.percentiles),
                    )
                ),
                "cdf": {
                    cutoff: (scores[source_name] < cutoff).sum() / num_predictions * 100
                    for cutoff in self.cutoffs[source_name]
                },
            }
            for source_name in scores.keys()
        }

        return response

    def batchbald_scores(self, predictions: NDArray[np.float_]) -> NDArray[np.float_]:
        if len(predictions.shape) != 3:
            raise ValueError(
                "To compute statistics for batchbald,"
                " `predictions` must be 3-dimensional,"
                f" but its {len(predictions.shape)} dimensions"
                f" are {predictions.shape}."
            )

        # Exclude samples we've been asked to exclude
        included_samples: NDArray[np.int_] = np.setdiff1d(
            np.arange(len(predictions)),
            self.batchbald_excluded_samples,
            assume_unique=False,
        )
        included_predictions: NDArray[np.float_] = predictions[included_samples, ...]

        # Convert prediction values to logarithms
        epsilon: float = 7.8886090522101180541e-31  # 2**-100
        included_predictions[included_predictions < epsilon] = epsilon
        log_predictions: NDArray[np.float_] = np.log(included_predictions)

        # Indicate how many predictions we want batchbald to rate as uncertain.  All
        # the remaining will be rated as more certain via a fallback approach.
        batch_size: int = min(self.batchbald_batch_size, len(log_predictions))
        num_samples: int = self.batchbald_num_samples

        # Do the BatchBALD calculation
        import torch
        import batchbald_redux as bbald
        import batchbald_redux.batchbald

        with torch.no_grad():
            bald: bbald.batchbald.CandidateBatch
            bald = bbald.batchbald.get_batchbald_batch(
                torch.from_numpy(log_predictions),
                batch_size,
                num_samples,
                dtype=torch.double,
            )
        bald_indices: NDArray[np.int_] = np.array(bald.indices)
        bald_scores: NDArray[np.float_] = np.array(bald.scores)

        # For samples that are not ranked by batchbald, we will fallback to using
        # (possibly shifted) confidence scores.  Predictions will be averaged over
        # Bayesian samples and then the most likely mean prediction score will be
        # the confidence.  These confidence scores are shifted, if needed, to ensure
        # that the confidences scores are interpretted as more certain than the
        # batchbald scores.  Because the user labels the most uncertain (lowest)
        # scores first, this puts the batchbald selections first, followed by the
        # remaining selections once the batchbald selections are exhausted.
        max_bald_scores: float = max(0.0, max(bald_scores))
        response: NDArray[np.float_] = np.full(predictions.shape[:-1], max_bald_scores)
        response += np.max(np.mean(predictions, axis=-2, keepdims=True), axis=-1)

        # Overwrite the fallback scores for the samples that are selected by batchbald.
        response[included_samples[bald_indices], :] = bald_scores[:, np.newaxis]
        return response
