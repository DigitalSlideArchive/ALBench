import numpy as np


class Strategy:
    def __init__():
        self.scoring_metric = None
        self.diversity_metric = None

    def set_scoring_metric(scoring_metric):
        self.scoring_metric = scoring_metric

    def get_scoring_metric(scoring_metric):
        return self.scoring_metric

    def set_diversity_metric(diversity_metric):
        self.diversity_metric = diversity_metric

    def get_diversity_metric(diversity_metric):
        return self.diversity_metric

    """
    Also support using Model *and its already computed predictions* to compute scores
    and/or to compute diversity!!!
    """
