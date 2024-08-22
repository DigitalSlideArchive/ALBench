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

from typing import Any, Mapping, Sequence

import numpy as np
import torch

import al_bench as alb
import al_bench.factory
from check import NDArrayFloat, NDArrayInt, deep_print, deeply_allclose
from create import create_dirichlet_predictions
from data_0050_factory import best_cutoffs, expected_certainties


def test_0050_factory() -> None:
    rng: np.random._generator.Generator = np.random.default_rng(seed=20240108)
    torch.manual_seed(20240102)

    # Gamma distribution has mean = alpha/beta and variance = alpha/beta^2.
    num_samples: int = 24
    num_repeats: int = 100
    num_classes: int = 3
    pseudocount_total_mean: float = 5.0
    pseudocount_total_variance: float = 4.0
    pseudocount_mean: float = pseudocount_total_mean / num_classes
    pseudocount_variance: float = pseudocount_total_variance / num_classes
    beta_hyperprior: float = pseudocount_mean / pseudocount_variance
    alpha_hyperprior: float = pseudocount_mean * beta_hyperprior
    sample_pseudocounts: NDArrayFloat
    predictions: NDArrayFloat
    sample_pseudocounts, predictions = create_dirichlet_predictions(
        num_samples=num_samples,
        num_repeats=num_repeats,
        num_classes=num_classes,
        alpha_hyperprior=alpha_hyperprior,
        beta_hyperprior=beta_hyperprior,
        rng=rng,
    )

    percentiles: NDArrayFloat = np.fromiter(range(0, 101, 10), dtype=float)
    compute_certainty: alb.factory.ComputeCertainty
    cutoffs: Mapping[str, Sequence[float]] = (
        best_cutoffs
        if len(best_cutoffs)
        else {t: [] for t in alb.factory.ComputeCertainty.all_certainty_types}
    )

    compute_certainty = alb.factory.ComputeCertainty(
        certainty_type=alb.factory.ComputeCertainty.all_certainty_types,
        percentiles=percentiles,
        cutoffs=cutoffs,
    )
    compute_certainty.set_batchbald_excluded_samples(np.array((0,)))
    certainties: Mapping[str, Mapping[str, Any]]
    certainties = compute_certainty.from_numpy_array(predictions)

    # Note that applying np.argsort twice produces integers in the same order as the
    # original input.
    batchbald_scores: NDArrayFloat = certainties["batchbald"]["scores"][:, 0]
    total_pseudocounts: NDArrayFloat = np.sum(sample_pseudocounts, axis=-1)
    batchbald_index: NDArrayInt = np.argsort(np.argsort(batchbald_scores))
    pseudocounts_index: NDArrayInt = np.argsort(np.argsort(total_pseudocounts))
    correlation: float = float(np.corrcoef(batchbald_index, pseudocounts_index)[0, 1])

    if True:
        print(f"{num_samples = }")
        print(f"{num_repeats = }")
        print(f"{num_classes = }")
        print(f"{alpha_hyperprior = }")
        print(f"{beta_hyperprior = }")
        print(f"{batchbald_scores = }")
        print(f"{total_pseudocounts = }")
        print(f"{batchbald_index = }")
        print(f"{pseudocounts_index = }")
        print(f"{correlation = }")
        # Torch is still inserting randomness, so we cannot expect batchbald to produce
        # deterministic results.  Instead simply check that the correlation is large
        # enough.  (Note that the threshold for large enough reasonably depends upon the
        # hyperpriors.)
        del certainties["batchbald"], expected_certainties["batchbald"]
        passed: bool = (
            deeply_allclose(certainties, expected_certainties) and correlation > 0.75
        )
        print(f"test_0050_factory() {'passed' if passed else 'failed'}")
        if not passed:
            raise ValueError("test_0050_factory() failed")
    else:
        # To regenerate the expected output, set this code section to run and, from a
        # shell prompt, run the following twice or thrice.  Then revert the file edit so
        # that this section is not run.
        """
        (
        python test_0050_factory.py > data_0050_factory2.py
        black -C data_0050_factory2.py
        mv data_0050_factory2.py data_0050_factory.py
        )
        """
        print("import numpy as np")
        print(f"expected_certainties = {deep_print(certainties)}")
        cutoffs = {
            k: list(v["percentiles"].values()) for k, v in expected_certainties.items()
        }
        print(f"best_cutoffs = {deep_print(cutoffs)}")


if __name__ == "__main__":
    test_0050_factory()
