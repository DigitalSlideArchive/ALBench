# ==========================================================================
#
#   Copyright NumFOCUS
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#          http://www.apache.org/licenses/LICENSE-2.0.txt
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ==========================================================================

import numpy as np

"""
StrategyHandler is a class that can be configured to run an active learning strategy.
At some point we should have a abstract superclass that defines the interface!!!
"""


class StrategyHandler:
    def __init__(self):
        self.scoring_metric = None
        self.diversity_metric = None

    def set_scoring_metric(self, scoring_metric):
        if scoring_metric is None:
            raise ValueError("The argument to set_scoring_metric must not be None")
        self.scoring_metric = scoring_metric

    def get_scoring_metric(self):
        return self.scoring_metric

    def clear_scoring_metric(self):
        self.scoring_metric = None

    def set_diversity_metric(self, diversity_metric):
        if diversity_metric is None:
            raise ValueError("The argument to set_diversity_metric must not be None")
        self.diversity_metric = diversity_metric

    def get_diversity_metric(self):
        return self.diversity_metric

    def clear_diversity_metric(self):
        self.diversity_metric = None

    """
    Also support using Model *and its already computed predictions* to compute scores
    and/or to compute diversity!!!
    """

    """
    Select new examples to be labeled by the expert.  `number_to_select` is the number
    that should be selected.  `currently_selected` is the list of examples that have
    already been selected, and their current labels.  `features` is the feature vector
    for each example in the entire set of examples, both those examples that have been
    labeled and those that could be selected for labeling.
    """

    def select_next_examples(self, number_to_select, currently_selected, features):
        # This implementation simply selects a subset of labels from those not currently
        # selected

        if number_to_select + len(currently_selected) > features.shape[0]:
            raise ValueError(
                f"Cannot not select {number_to_select} unlabeled feature vectors."
            )

        import random

        return random.sample(
            [x for x in range(features.shape[0]) if x not in currently_selected],
            number_to_select,
        )
