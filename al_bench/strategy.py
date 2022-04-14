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
        self.model_handler = None
        self.dataset_handler = None
        self.scoring_metric = None
        self.diversity_metric = None

    def set_model_handler(self, model_handler):
        if model_handler is None:
            raise ValueError("The argument to set_model_handler must not be None")
        self.model_handler = model_handler

    def get_model_handler(self):
        return self.model_handler

    def clear_model_handler(self):
        self.model_handler = None

    def set_dataset_handler(self, dataset_handler):
        if dataset_handler is None:
            raise ValueError("The argument to set_dataset_handler must not be None")
        self.dataset_handler = dataset_handler

    def get_dataset_handler(self):
        return self.dataset_handler

    def clear_dataset_handler(self):
        self.dataset_handler = None

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
        pass
