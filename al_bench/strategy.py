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

import numpy as np
import scipy.stats
from . import dataset, model


class AbstractScoringMetric:
    def __init__(self):
        raise NotImplementedError(
            "Abstract method AbstractScoringMetric::__init__ should not be called."
        )


class AbstractDiversityMetric:
    def __init__(self):
        raise NotImplementedError(
            "Abstract method AbstractDiversityMetric::__init__ should not be called."
        )


class AbstractStrategyHandler:
    """
    AbstractStrategyHandler is an abstract class that defines the interface for an
    active learning strategy.
    """

    def __init__(self):
        raise NotImplementedError(
            "Abstract method AbstractStrategyHandler::__init__ should not be called."
        )

    def set_dataset_handler(self, dataset_handler):
        raise NotImplementedError(
            "Abstract method AbstractStrategyHandler::set_dataset_handler "
            "should not be called."
        )

    def get_dataset_handler(self):
        raise NotImplementedError(
            "Abstract method AbstractStrategyHandler::get_dataset_handler "
            "should not be called."
        )

    def set_model_handler(self, model_handler):
        raise NotImplementedError(
            "Abstract method AbstractStrategyHandler::set_model_handler "
            "should not be called."
        )

    def get_model_handler(self):
        raise NotImplementedError(
            "Abstract method AbstractStrategyHandler::get_model_handler "
            "should not be called."
        )

    def set_desired_outputs(self):
        raise NotImplementedError(
            "Abstract method AbstractStrategyHandler::set_desired_outputs "
            "should not be called."
        )

    def set_scoring_metric(self, scoring_metric):
        raise NotImplementedError(
            "Abstract method AbstractStrategyHandler::set_scoring_metric "
            "should not be called."
        )

    def get_scoring_metric(self):
        raise NotImplementedError(
            "Abstract method AbstractStrategyHandler::get_scoring_metric "
            "should not be called."
        )

    def clear_scoring_metric(self):
        raise NotImplementedError(
            "Abstract method AbstractStrategyHandler::clear_scoring_metric "
            "should not be called."
        )

    def set_diversity_metric(self, diversity_metric):
        raise NotImplementedError(
            "Abstract method AbstractStrategyHandler::set_diversity_metric "
            "should not be called."
        )

    def get_diversity_metric(self):
        raise NotImplementedError(
            "Abstract method AbstractStrategyHandler::get_diversity_metric "
            "should not be called."
        )

    def clear_diversity_metric(self):
        raise NotImplementedError(
            "Abstract method AbstractStrategyHandler::clear_diversity_metric "
            "should not be called."
        )

    def set_learning_parameters(self, **parameters):
        raise NotImplementedError(
            "Abstract method AbstractStrategyHandler::set_learning_parameters "
            "should not be called."
        )

    def get_learning_parameters(self):
        raise NotImplementedError(
            "Abstract method AbstractStrategyHandler::get_learning_parameters "
            "should not be called."
        )

    def select_next_examples(self, currently_labeled_examples, validation_indices=None):
        raise NotImplementedError(
            "Abstract method AbstractStrategyHandler::select_next_examples "
            "should not be called."
        )

    def run(self, currently_labeled_examples):
        raise NotImplementedError(
            "Abstract method AbstractStrategyHandler::run should not be called."
        )


class GenericStrategyHandler(AbstractStrategyHandler):
    """
    GenericStrategyHandler handles functionality that is agnostic to the choice of the
    active learning strategy.  GenericStrategyHandler is the superclass for
    RandomStrategyHandler, LeastConfidenceStrategyHandler, LeastMarginStrategyHandler,
    EntropyStrategyHandler, etc.  These subclasses handle AbstractStrategyHandler
    operations that are dependent upon which active learning strategy is being used.
    """

    def __init__(self):
        # super(GenericStrategyHandler, self).__init__()
        self.scoring_metric = None
        self.diversity_metric = None
        self.dataset_handler = None
        self.model_handler = None
        self.parameters = None

        # These parameters must be supplied to the set_learning_parameters call.
        self.required_parameters_keys = [
            "label_of_interest",
            "maximum_iterations",
            "number_to_select_per_iteration",
        ]
        # This the exhaustive list of parameters that are valid when used with a
        # set_learning_parameters call.
        self.valid_parameters_keys = self.required_parameters_keys + list()

    def set_dataset_handler(self, dataset_handler):
        if not isinstance(dataset_handler, dataset.AbstractDatasetHandler):
            raise ValueError(
                "The argument to set_dataset_handler must be a (subclass of) "
                "AbstractDatasetHandler"
            )
        self.dataset_handler = dataset_handler

    def get_dataset_handler(self):
        return self.dataset_handler

    def set_model_handler(self, model_handler):
        if not isinstance(model_handler, model.AbstractModelHandler):
            raise ValueError(
                "The argument to set_model_handler must be a (subclass of) "
                "AbstractModelHandler"
            )
        self.model_handler = model_handler

    def get_model_handler(self):
        return self.model_handler

    def set_desired_outputs(self):
        """
        Choose what statistics and other outputs should be recorded during the active
        learning strategy, for use in evaluating the combination of strategy, model, and
        data set.  This will likely be called from subclass overrides, much in the
        spirit of ITK's PrintSelf methods
        """
        # Alternatively, record everything and then have a method that selectively
        # fetches it.
        raise NotImplementedError("Not implemented")

    def set_scoring_metric(self, scoring_metric):
        """
        Supply a scoring metric that evaluates each unlabeled feature vector
        (individually) for how useful it would be if only its label were known.  A
        higher score indicates a higher usability.  The scoring_metric will be a
        subclass of an abstract AbstractScoringMetric class, and supports a specified
        interface.
        """
        if not isinstance(scoring_metric, AbstractScoringMetric):
            raise ValueError(
                "The argument to set_scoring_metric must be a subclass of "
                "AbstractScoringMetric"
            )
        self.scoring_metric = scoring_metric

    def get_scoring_metric(self):
        """
        Retrieve a previously supplied scoring metric
        """
        return self.scoring_metric

    def clear_scoring_metric(self):
        """
        Remove a previously supplied scoring metric
        """
        self.scoring_metric = None

    def set_diversity_metric(self, diversity_metric):
        """
        Supply a diversity metric that evaluates a set of feature vectors for their
        diversity.  A higher score indicates that the feature vectors are diverse.  The
        diversity_metric will be a subclass of an abstract AbstractDiversityMetric
        class, which supports a specified interface.
        """
        if not isinstance(diversity_metric, AbstractDiversityMetric):
            raise ValueError(
                "The argument to set_diversity_metric must be a subclass of "
                "AbstractDiversityMetric"
            )
        self.diversity_metric = diversity_metric

    def get_diversity_metric(self):
        """
        Retrieve a previously supplied diversity metric
        """
        return self.diversity_metric

    def clear_diversity_metric(self):
        """
        Remove a previously supplied diversity metric
        """
        self.diversity_metric = None

    def set_learning_parameters(self, **parameters):
        """
        If this strategy has parameters other than a scoring metric or diversity metric,
        set them here.
        """

        if not isinstance(parameters, dict):
            raise ValueError(
                f"The argument to set_learning_parameters must be (a subclass of) a "
                f"Python dict but is of type {type(parameters)}"
            )

        missing_keys = set(self.required_parameters_keys).difference(parameters.keys())
        if len(missing_keys) > 0:
            raise ValueError(
                f"set_learning_parameters missing required key(s): {missing_keys}"
            )

        invalid_keys = set(parameters.keys()).difference(self.valid_parameters_keys)
        if len(invalid_keys) > 0:
            raise ValueError(
                f"set_learning_parameters given invalid key(s): {invalid_keys}"
            )
        self.parameters = parameters

    def get_learning_parameters(self):
        return self.parameters

    def select_next_examples(self, currently_labeled_examples, validation_indices=None):
        raise NotImplementedError(
            "Abstract method GenericStrategyHandler::select_next_examples should not "
            "be called."
        )

    def reset_log(self):
        self.model_handler.reset_log()

    def get_log(self):
        return self.model_handler.get_log()

    def write_train_log_to_tensorboard_file(self, *args, **kwargs):
        return self.model_handler.write_train_log_to_tensorboard_file(*args, **kwargs)

    def write_epoch_log_to_tensorboard_file(self, *args, **kwargs):
        return self.model_handler.write_epoch_log_to_tensorboard_file(*args, **kwargs)

    def run(self, currently_labeled_examples):
        """
        Run the strategy, start to finish.
        """
        currently_labeled_examples = set(currently_labeled_examples)
        feature_vectors = self.dataset_handler.get_all_feature_vectors()
        labels = self.dataset_handler.get_all_labels()
        if len(labels.shape) == 1:
            labels = labels[:, np.newaxis]

        validation_indices = self.dataset_handler.get_validation_indices()
        validation_feature_vectors = (
            self.dataset_handler.get_validation_feature_vectors()
        )
        validation_labels = self.dataset_handler.get_validation_labels()
        assert (validation_indices is None) == (validation_feature_vectors is None)
        assert (validation_indices is None) == (validation_labels is None)
        if validation_labels is not None and len(validation_labels.shape) == 1:
            validation_labels = validation_labels[:, np.newaxis]
        validation_args = (
            list()
            if validation_feature_vectors is None
            else [validation_feature_vectors, validation_labels]
        )

        maximum_iterations = self.parameters["maximum_iterations"]
        label_of_interest = self.parameters["label_of_interest"]

        for iteration in range(maximum_iterations):
            print(f"Predicting for {feature_vectors.shape[0]} examples")
            self.predictions = self.model_handler.predict(feature_vectors)
            next_examples = self.select_next_examples(
                currently_labeled_examples, validation_indices
            )
            currently_labeled_examples = currently_labeled_examples.union(next_examples)
            current_indices = tuple(currently_labeled_examples)
            current_feature_vectors = feature_vectors[current_indices, :]
            current_labels = labels[current_indices, :]
            print(f"Training with {current_feature_vectors.shape[0]} examples")
            self.model_handler.train(
                current_feature_vectors,
                current_labels[:, label_of_interest],
                *validation_args,
            )

        print(f"Predicting for {feature_vectors.shape[0]} examples")
        self.predictions = self.model_handler.predict(feature_vectors)
        self.labeled_examples = currently_labeled_examples


class RandomStrategyHandler(GenericStrategyHandler):
    def __init__(self):
        super(RandomStrategyHandler, self).__init__()

    def select_next_examples(self, currently_labeled_examples, validation_indices=None):
        """
        Select new examples to be labeled by the expert.
        `number_to_select_per_iteration` is the number that should be selected with each
        iteration of the active learning strategy.  `currently_labeled_examples` is the
        list of examples indices that have already been labeled.  `feature_vectors` is
        the feature vector for each example in the entire set of examples, including
        both those examples that have been labeled and those that could be selected for
        labeling.
        """
        number_to_select = self.parameters["number_to_select_per_iteration"]
        feature_vectors = self.dataset_handler.get_all_feature_vectors()

        # Make sure the pool to select from is large enough
        if validation_indices is not None:
            currently_labeled_examples = set(currently_labeled_examples).union(
                set(validation_indices)
            )
        if (
            number_to_select + len(currently_labeled_examples)
            > feature_vectors.shape[0]
        ):
            raise ValueError(
                f"Cannot not select {number_to_select} unlabeled feature vectors; only "
                f"{feature_vectors.shape[0] - len(currently_labeled_examples)} remain."
            )

        # This implementation simply selects a random subset of labels from those not
        # currently selected.
        import random

        return random.sample(
            [
                example_index
                for example_index in range(feature_vectors.shape[0])
                if example_index not in currently_labeled_examples
            ],
            number_to_select,
        )


class LeastConfidenceStrategyHandler(GenericStrategyHandler):
    def __init__(self):
        super(LeastConfidenceStrategyHandler, self).__init__()

    def select_next_examples(self, currently_labeled_examples, validation_indices=None):
        """
        Select new examples to be labeled by the expert.  This choses the unlabeled
        examples with the smallest maximum score.
        """
        number_to_select = self.parameters["number_to_select_per_iteration"]
        number_of_feature_vectors = (
            self.dataset_handler.get_all_feature_vectors().shape[0]
        )

        # Make sure the pool to select from is large enough
        if validation_indices is not None:
            currently_labeled_examples = set(currently_labeled_examples).union(
                set(validation_indices)
            )
        if (
            number_to_select + len(currently_labeled_examples)
            > number_of_feature_vectors
        ):
            raise ValueError(
                f"Cannot not select {number_to_select} unlabeled feature vectors; only "
                f"{number_of_feature_vectors - len(currently_labeled_examples)} remain."
            )

        predictions = self.predictions
        # We assume that via "softmax" or similar, the values are already non-negative
        # and sum to 1.0.
        #   predictions = predictions / predictions.sum(axis=-1, keepdims=True)
        # For each example, how strong is the best category's score?
        predict_score = np.amax(predictions, axis=-1)
        # Make the currently labeled examples look good, so that they won't be selected
        if len(currently_labeled_examples):
            predict_score[np.array(list(currently_labeled_examples))] = 2
        # Find the lowest scoring examples
        predict_order = np.argsort(predict_score)[0:number_to_select]
        return predict_order


class LeastMarginStrategyHandler(GenericStrategyHandler):
    def __init__(self):
        super(LeastMarginStrategyHandler, self).__init__()

    def select_next_examples(self, currently_labeled_examples, validation_indices=None):
        """
        Select new examples to be labeled by the expert.  This choses the unlabeled
        examples with the smallest gap between highest and second-highest score.
        """
        number_to_select = self.parameters["number_to_select_per_iteration"]
        number_of_feature_vectors = (
            self.dataset_handler.get_all_feature_vectors().shape[0]
        )

        # Make sure the pool to select from is large enough
        if validation_indices is not None:
            currently_labeled_examples = set(currently_labeled_examples).union(
                set(validation_indices)
            )
        if (
            number_to_select + len(currently_labeled_examples)
            > number_of_feature_vectors
        ):
            raise ValueError(
                f"Cannot not select {number_to_select} unlabeled feature vectors; only "
                f"{number_of_feature_vectors - len(currently_labeled_examples)} remain."
            )

        predictions = self.predictions
        # We assume that via "softmax" or similar, the values are already non-negative
        # and sum to 1.0.
        #   predictions = predictions / predictions.sum(axis=-1, keepdims=True)
        # Find the largest and second largest values, and compute their difference
        predict_indices = np.arange(len(predictions))
        predict_argsort = np.argsort(predictions, axis=-1)
        predict_score = (
            predictions[predict_indices, predict_argsort[:, -1]]
            - predictions[predict_indices, predict_argsort[:, -2]]
        )
        # Make the currently labeled examples look good, so that they won't be selected
        if len(currently_labeled_examples):
            predict_score[np.array(list(currently_labeled_examples))] = 2
        # Find the lowest scoring examples
        predict_order = np.argsort(predict_score)[0:number_to_select]
        return predict_order


class EntropyStrategyHandler(GenericStrategyHandler):
    def __init__(self):
        super(EntropyStrategyHandler, self).__init__()

    def select_next_examples(self, currently_labeled_examples, validation_indices=None):
        """
        Select new examples to be labeled by the expert.  This choses the unlabeled
        examples with the highest entropy.
        """
        number_to_select = self.parameters["number_to_select_per_iteration"]
        number_of_feature_vectors = (
            self.dataset_handler.get_all_feature_vectors().shape[0]
        )

        # Make sure the pool to select from is large enough
        if validation_indices is not None:
            currently_labeled_examples = set(currently_labeled_examples).union(
                set(validation_indices)
            )
        if (
            number_to_select + len(currently_labeled_examples)
            > number_of_feature_vectors
        ):
            raise ValueError(
                f"Cannot not select {number_to_select} unlabeled feature vectors; only "
                f"{number_of_feature_vectors - len(currently_labeled_examples)} remain."
            )

        predictions = self.predictions
        # We assume that via "softmax" or similar, the values are already non-negative
        # and sum to 1.0.
        #   predictions = predictions / predictions.sum(axis=-1, keepdims=True)
        # Scipy will (normalize row values to sum to 1 and then) compute the entropy of
        # each row.  We negate the entropy so that the distributions that are near
        # uniform have the smallest (i.e., most negative) values.
        predict_score = -scipy.stats.entropy(predictions, axis=-1)
        # Make the currently labeled examples look good, so that they won't be selected
        if len(currently_labeled_examples):
            predict_score[np.array(list(currently_labeled_examples))] = 2
        # Find the lowest scoring examples
        predict_order = np.argsort(predict_score)[0:number_to_select]
        return predict_order
