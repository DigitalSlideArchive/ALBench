"""Active Learning Benchmarking tool"""

__version__ = "0.0.1"

"""

The tool takes an input dataset, machine learning model, and active learning strategy
and outputs information to be used in evaluating how well the strategy does with that
model and dataset.  By running the tool multiple times with different inputs, the tool
allows comparisons across different active learning strategies and also allows
comparisons across different models and across different datasets.  Researchers can use
the tool to test proposed active learning strategies in the context of a specific model
and dataset; or multiple models and datasets can be used to get a broader picture of
each strategy's effectiveness in multiple contexts.  As an alternative use case,
multiple runs of the tool with different models and datasets can be compared, evaluating
these models and datasets for their compatibility with a given active learning strategy.

"""
