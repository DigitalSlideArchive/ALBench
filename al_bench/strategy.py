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
        self.model_handler = model_handler

    def get_model_handler(self):
        return self.model_handler

    def set_dataset_handler(self, dataset_handler):
        self.dataset_handler = dataset_handler

    def get_dataset_handler(self):
        return self.dataset_handler

    def set_scoring_metric(self, scoring_metric):
        self.scoring_metric = scoring_metric

    def get_scoring_metric(self):
        return self.scoring_metric

    def set_diversity_metric(self, diversity_metric):
        self.diversity_metric = diversity_metric

    def get_diversity_metric(self):
        return self.diversity_metric

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
