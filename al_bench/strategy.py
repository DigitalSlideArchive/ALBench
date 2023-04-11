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
from typing import List, Mapping, Set
from . import dataset, model


class AbstractScoringMetric:
    def __init__(self) -> None:
        raise NotImplementedError(
            "Abstract method AbstractScoringMetric::__init__ should not be called."
        )


class AbstractDiversityMetric:
    def __init__(self) -> None:
        raise NotImplementedError(
            "Abstract method AbstractDiversityMetric::__init__ should not be called."
        )


class AbstractStrategyHandler:
    """
    AbstractStrategyHandler is an abstract class that defines the interface for an
    active learning strategy.
    """

    def __init__(self) -> None:
        raise NotImplementedError(
            "Abstract method AbstractStrategyHandler::__init__ should not be called."
        )

    def set_dataset_handler(
        self, dataset_handler: dataset.AbstractDatasetHandler
    ) -> None:
        raise NotImplementedError(
            "Abstract method AbstractStrategyHandler::set_dataset_handler"
            " should not be called."
        )

    def get_dataset_handler(self) -> dataset.AbstractDatasetHandler:
        raise NotImplementedError(
            "Abstract method AbstractStrategyHandler::get_dataset_handler"
            " should not be called."
        )

    def set_model_handler(self, model_handler: model.AbstractModelHandler) -> None:
        raise NotImplementedError(
            "Abstract method AbstractStrategyHandler::set_model_handler"
            " should not be called."
        )

    def get_model_handler(self) -> model.AbstractModelHandler:
        raise NotImplementedError(
            "Abstract method AbstractStrategyHandler::get_model_handler"
            " should not be called."
        )

    def set_desired_outputs(self) -> None:
        raise NotImplementedError(
            "Abstract method AbstractStrategyHandler::set_desired_outputs"
            " should not be called."
        )

    def set_scoring_metric(self, scoring_metric: AbstractScoringMetric) -> None:
        raise NotImplementedError(
            "Abstract method AbstractStrategyHandler::set_scoring_metric"
            " should not be called."
        )

    def get_scoring_metric(self) -> AbstractScoringMetric:
        raise NotImplementedError(
            "Abstract method AbstractStrategyHandler::get_scoring_metric"
            " should not be called."
        )

    def clear_scoring_metric(self) -> None:
        raise NotImplementedError(
            "Abstract method AbstractStrategyHandler::clear_scoring_metric"
            " should not be called."
        )

    def set_diversity_metric(self, diversity_metric: AbstractDiversityMetric) -> None:
        raise NotImplementedError(
            "Abstract method AbstractStrategyHandler::set_diversity_metric"
            " should not be called."
        )

    def get_diversity_metric(self) -> AbstractDiversityMetric:
        raise NotImplementedError(
            "Abstract method AbstractStrategyHandler::get_diversity_metric"
            " should not be called."
        )

    def clear_diversity_metric(self) -> None:
        raise NotImplementedError(
            "Abstract method AbstractStrategyHandler::clear_diversity_metric"
            " should not be called."
        )

    def set_learning_parameters(self, **parameters) -> None:
        raise NotImplementedError(
            "Abstract method AbstractStrategyHandler::set_learning_parameters"
            " should not be called."
        )

    def get_learning_parameters(self) -> Mapping:
        raise NotImplementedError(
            "Abstract method AbstractStrategyHandler::get_learning_parameters"
            " should not be called."
        )

    def select_next_indices(
        self,
        labeled_indices: NDArray,
        validation_indices: NDArray = np.array((), dtype=np.int64),
    ) -> NDArray:
        raise NotImplementedError(
            "Abstract method AbstractStrategyHandler::select_next_indices"
            " should not be called."
        )

    def reset_log(self) -> None:
        raise NotImplementedError(
            "Abstract method AbstractStrategyHandler::reset_log should not be called."
        )

    def get_log(self) -> List:
        raise NotImplementedError(
            "Abstract method AbstractStrategyHandler::get_log should not be called."
        )

    def write_train_log_for_tensorboard(self, *args, **kwargs) -> bool:
        raise NotImplementedError(
            "Abstract method AbstractStrategyHandler::write_train_log_for_tensorboard"
            " should not be called."
        )

    def write_epoch_log_for_tensorboard(self, *args, **kwargs) -> bool:
        raise NotImplementedError(
            "Abstract method AbstractStrategyHandler::write_epoch_log_for_tensorboard"
            " should not be called."
        )

    def write_confidence_log_for_tensorboard(self, *args, **kwargs) -> bool:
        raise NotImplementedError(
            "Abstract method AbstractStrategyHandler::write_confidence_log_for"
            "_tensorboard should not be called."
        )

    def run(self, labeled_indices: NDArray) -> None:
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

    def __init__(self) -> None:
        # super(GenericStrategyHandler, self).__init__()
        pass

        # These parameters must be supplied to the set_learning_parameters call.
        self.required_parameters_keys = [
            "label_of_interest",
            "maximum_queries",
            "number_to_select_per_query",
        ]
        # This the exhaustive list of parameters that are valid when used with a
        # set_learning_parameters call.
        self.valid_parameters_keys = self.required_parameters_keys + list()

    def set_dataset_handler(
        self, dataset_handler: dataset.AbstractDatasetHandler
    ) -> None:
        if not isinstance(dataset_handler, dataset.AbstractDatasetHandler):
            raise ValueError(
                "The argument to set_dataset_handler must be a (subclass of)"
                " AbstractDatasetHandler"
            )
        self.dataset_handler = dataset_handler

    def get_dataset_handler(self) -> dataset.AbstractDatasetHandler:
        return self.dataset_handler

    def set_model_handler(self, model_handler: model.AbstractModelHandler) -> None:
        if not isinstance(model_handler, model.AbstractModelHandler):
            raise ValueError(
                "The argument to set_model_handler must be a (subclass of)"
                " AbstractModelHandler"
            )
        self.model_handler = model_handler

    def get_model_handler(self) -> model.AbstractModelHandler:
        return self.model_handler

    def set_desired_outputs(self) -> None:
        """
        Choose what statistics and other outputs should be recorded during the active
        learning strategy, for use in evaluating the combination of strategy, model, and
        data set.  This will likely be called from subclass overrides, much in the
        spirit of ITK's PrintSelf methods
        """
        # Alternatively, record everything and then have a method that selectively
        # fetches it.
        raise NotImplementedError("Not implemented")

    def set_scoring_metric(self, scoring_metric: AbstractScoringMetric) -> None:
        """
        Supply a scoring metric that evaluates each unlabeled feature vector
        (individually) for how useful it would be if only its label were known.  A
        higher score indicates a higher usability.  The scoring_metric will be a
        subclass of an abstract AbstractScoringMetric class, and supports a specified
        interface.
        """
        if not isinstance(scoring_metric, AbstractScoringMetric):
            raise ValueError(
                "The argument to set_scoring_metric must be a subclass of"
                " AbstractScoringMetric"
            )
        self.scoring_metric = scoring_metric

    def get_scoring_metric(self) -> AbstractScoringMetric:
        """
        Retrieve a previously supplied scoring metric
        """
        return self.scoring_metric

    def clear_scoring_metric(self) -> None:
        """
        Remove a previously supplied scoring metric
        """
        del self.scoring_metric

    def set_diversity_metric(self, diversity_metric: AbstractDiversityMetric) -> None:
        """
        Supply a diversity metric that evaluates a set of feature vectors for their
        diversity.  A higher score indicates that the feature vectors are diverse.  The
        diversity_metric will be a subclass of an abstract AbstractDiversityMetric
        class, which supports a specified interface.
        """
        if not isinstance(diversity_metric, AbstractDiversityMetric):
            raise ValueError(
                "The argument to set_diversity_metric must be a subclass of"
                " AbstractDiversityMetric"
            )
        self.diversity_metric = diversity_metric

    def get_diversity_metric(self) -> AbstractDiversityMetric:
        """
        Retrieve a previously supplied diversity metric
        """
        return self.diversity_metric

    def clear_diversity_metric(self) -> None:
        """
        Remove a previously supplied diversity metric
        """
        del self.diversity_metric

    def set_learning_parameters(self, **parameters) -> None:
        """
        If this strategy has parameters other than a scoring metric or diversity metric,
        set them here.
        """

        if not isinstance(parameters, dict):
            raise ValueError(
                f"The argument to set_learning_parameters must be (a subclass of) a"
                f" Python dict but is of type {type(parameters)}"
            )

        missing_keys: Set = set(self.required_parameters_keys) - set(parameters)
        if len(missing_keys) > 0:
            raise ValueError(
                f"set_learning_parameters missing required key(s): {missing_keys}"
            )

        invalid_keys: Set = set(parameters) - set(self.valid_parameters_keys)
        if len(invalid_keys) > 0:
            raise ValueError(
                f"set_learning_parameters given invalid key(s): {invalid_keys}"
            )
        self.parameters = parameters

    def get_learning_parameters(self) -> Mapping:
        return self.parameters

    def select_next_indices(
        self,
        labeled_indices: NDArray,
        validation_indices: NDArray = np.array((), dtype=np.int64),
    ) -> NDArray:
        raise NotImplementedError(
            "Abstract method GenericStrategyHandler::select_next_indices should not"
            " be called."
        )

    def reset_log(self) -> None:
        self.model_handler.reset_log()

    def get_log(self) -> List:
        return self.model_handler.get_log()

    def write_train_log_for_tensorboard(self, *args, **kwargs) -> bool:
        return self.model_handler.write_train_log_for_tensorboard(*args, **kwargs)

    def write_epoch_log_for_tensorboard(self, *args, **kwargs) -> bool:
        return self.model_handler.write_epoch_log_for_tensorboard(*args, **kwargs)

    def write_confidence_log_for_tensorboard(self, *args, **kwargs) -> bool:
        return self.model_handler.write_confidence_log_for_tensorboard(*args, **kwargs)

    def run(self, labeled_indices: NDArray) -> None:
        """
        Run the strategy, start to finish.
        """
        # Should we compute uncertainty for all predictions or just the predictions for
        # unlabeled examples?
        all_p: bool = False

        feature_vectors: NDArray = self.dataset_handler.get_all_feature_vectors()
        labels: NDArray = self.dataset_handler.get_all_labels()
        if len(labels.shape) == 1:
            labels = labels[:, np.newaxis]

        validation_indices: NDArray = self.dataset_handler.get_validation_indices()
        validation_feature_vectors: NDArray = (
            self.dataset_handler.get_validation_feature_vectors()
        )
        validation_labels: NDArray = self.dataset_handler.get_validation_labels()
        if len(validation_labels.shape) == 1:
            validation_labels = validation_labels[:, np.newaxis]
        validation_args: List = (
            list()
            if len(validation_feature_vectors.shape) == 0
            else [validation_feature_vectors, validation_labels]
        )

        maximum_queries: int = self.parameters["maximum_queries"]
        label_of_interest: int = self.parameters["label_of_interest"]

        # Do initial training
        self.model_handler.reinitialize_weights()
        current_feature_vectors: NDArray = feature_vectors[labeled_indices, :]
        current_labels: NDArray = labels[labeled_indices, label_of_interest]
        if len(current_feature_vectors) > 0:
            print(f"Training with {current_feature_vectors.shape[0]} examples")
            self.model_handler.train(
                current_feature_vectors, current_labels, *validation_args
            )
        # Loop through the queries
        all_indices: Set = set(range(len(feature_vectors)))
        for query in range(maximum_queries):
            # Evaluate the pool of possibilities
            print(f"Predicting for {feature_vectors.shape[0]} examples")
            self.predictions = self.model_handler.predict(feature_vectors, log_it=all_p)
            if not all_p:
                # Log the predictions for the unlabeled examples only
                unlabeled_indices: Set = all_indices - set(labeled_indices)
                unlabeled_predictions: NDArray = self.predictions[
                    np.fromiter(unlabeled_indices, dtype=np.int64)
                ]
                self.model_handler.get_logger().on_predict_end(
                    {"outputs": unlabeled_predictions}
                )
            # Find the next indices to be labeled
            next_indices: NDArray = self.select_next_indices(
                labeled_indices, validation_indices
            )
            # Query the oracle to get labels for the next_indices.  This call returns
            # all available labels, old and new.
            labels = self.dataset_handler.query_oracle(next_indices)
            if len(labels.shape) == 1:
                labels = labels[:, np.newaxis]
            # Update the list of labeled_indices
            labeled_indices = np.fromiter(
                set(labeled_indices) | set(next_indices), dtype=np.int64
            )
            # Do training with the update list of labeled_indices
            current_feature_vectors = feature_vectors[labeled_indices, :]
            current_labels = labels[labeled_indices, label_of_interest]
            print(f"Training with {current_feature_vectors.shape[0]} examples")
            self.model_handler.train(
                current_feature_vectors, current_labels, *validation_args
            )

        print(f"Predicting for {feature_vectors.shape[0]} examples")
        self.predictions = self.model_handler.predict(feature_vectors, log_it=all_p)
        if not all_p:
            # Log the predictions for the unlabeled examples only
            unlabeled_indices = all_indices - set(labeled_indices)
            unlabeled_predictions = self.predictions[
                np.fromiter(unlabeled_indices, dtype=np.int64)
            ]
            self.model_handler.logger.on_predict_end({"outputs": unlabeled_predictions})
        self.labeled_indices: NDArray = labeled_indices


class RandomStrategyHandler(GenericStrategyHandler):
    def __init__(self) -> None:
        super(RandomStrategyHandler, self).__init__()

    def select_next_indices(
        self,
        labeled_indices: NDArray,
        validation_indices: NDArray = np.array((), dtype=np.int64),
    ) -> NDArray:
        """
        Select new examples to be labeled by the expert.
        `number_to_select_per_query` is the number that should be selected with each
          query of the active learning strategy.
        `labeled_indices` is the set of examples indices that have already been labeled.
        `feature_vectors` is the feature vector for each example in the entire set of
          examples, including both those examples that have been labeled and those that
          could be selected for labeling.
        """
        number_to_select: int = self.parameters["number_to_select_per_query"]
        feature_vectors: NDArray = self.dataset_handler.get_all_feature_vectors()

        # Make sure the pool to select from is large enough
        excluded_indices: Set = set(labeled_indices) | set(validation_indices)
        if number_to_select + len(excluded_indices) > feature_vectors.shape[0]:
            raise ValueError(
                f"Cannot not select {number_to_select} unlabeled feature vectors; only"
                f" {feature_vectors.shape[0] - len(excluded_indices)} remain."
            )

        # This implementation simply selects a random subset of labels from those not
        # currently selected.
        import random

        return np.array(
            random.sample(
                [
                    example_index
                    for example_index in range(feature_vectors.shape[0])
                    if example_index not in excluded_indices
                ],
                number_to_select,
            )
        )


class LeastConfidenceStrategyHandler(GenericStrategyHandler):
    def __init__(self) -> None:
        super(LeastConfidenceStrategyHandler, self).__init__()

    def select_next_indices(
        self,
        labeled_indices: NDArray,
        validation_indices: NDArray = np.array((), dtype=np.int64),
    ) -> NDArray:
        """
        Select new examples to be labeled by the expert.  This choses the unlabeled
        examples with the smallest maximum score.
        """
        number_to_select: int = self.parameters["number_to_select_per_query"]
        number_of_feature_vectors: int = (
            self.dataset_handler.get_all_feature_vectors().shape[0]
        )

        # Make sure the pool to select from is large enough
        excluded_indices: Set = set(labeled_indices) | set(validation_indices)
        if number_to_select + len(excluded_indices) > number_of_feature_vectors:
            raise ValueError(
                f"Cannot not select {number_to_select} unlabeled feature vectors; only"
                f" {number_of_feature_vectors - len(excluded_indices)} remain."
            )

        predictions: NDArray = self.predictions
        if np.amax(predictions) <= 0.0:
            # Convert log_softmax to softmax
            predictions = np.exp(predictions)
        # We assume that via "softmax" or similar, the values are already non-negative.
        # Probably they also already sum to 1.0, but let's force that anyway.
        predictions = predictions / predictions.sum(axis=-1, keepdims=True)
        # For each example, how strong is the best category's score?
        predict_score = np.amax(predictions, axis=-1)
        # Make the currently labeled examples look confident, so that they won't be
        # selected
        if len(excluded_indices):
            predict_score[np.fromiter(excluded_indices, dtype=np.int64)] = 2
        # Find the lowest scoring examples
        predict_order: NDArray = np.argsort(predict_score)[0:number_to_select]
        return predict_order


class LeastMarginStrategyHandler(GenericStrategyHandler):
    def __init__(self) -> None:
        super(LeastMarginStrategyHandler, self).__init__()

    def select_next_indices(
        self,
        labeled_indices: NDArray,
        validation_indices: NDArray = np.array((), dtype=np.int64),
    ) -> NDArray:
        """
        Select new examples to be labeled by the expert.  This choses the unlabeled
        examples with the smallest gap between highest and second-highest score.
        """
        number_to_select: int = self.parameters["number_to_select_per_query"]
        number_of_feature_vectors: int = (
            self.dataset_handler.get_all_feature_vectors().shape[0]
        )

        # Make sure the pool to select from is large enough
        excluded_indices: Set = set(labeled_indices) | set(validation_indices)
        if number_to_select + len(excluded_indices) > number_of_feature_vectors:
            raise ValueError(
                f"Cannot not select {number_to_select} unlabeled feature vectors; only"
                f" {number_of_feature_vectors - len(excluded_indices)} remain."
            )

        predictions: NDArray = self.predictions
        if np.amax(predictions) <= 0.0:
            # Convert log_softmax to softmax
            predictions = np.exp(predictions)
        # We assume that via "softmax" or similar, the values are already non-negative.
        # Probably they also already sum to 1.0, but let's force that anyway.
        predictions = predictions / predictions.sum(axis=-1, keepdims=True)
        # Find the largest and second largest values, and compute their difference
        predict_indices: NDArray = np.arange(len(predictions))
        predict_argsort: NDArray = np.argsort(predictions, axis=-1)
        predict_score: NDArray = (
            predictions[predict_indices, predict_argsort[:, -1]]
            - predictions[predict_indices, predict_argsort[:, -2]]
        )
        # Make the currently labeled examples look confident, so that they won't be
        # selected
        if len(excluded_indices):
            predict_score[np.fromiter(excluded_indices, dtype=np.int64)] = 2
        # Find the lowest scoring examples
        predict_order: NDArray = np.argsort(predict_score)[0:number_to_select]
        return predict_order


class EntropyStrategyHandler(GenericStrategyHandler):
    def __init__(self) -> None:
        super(EntropyStrategyHandler, self).__init__()

    def select_next_indices(
        self,
        labeled_indices: NDArray,
        validation_indices: NDArray = np.array((), dtype=np.int64),
    ) -> NDArray:
        """
        Select new examples to be labeled by the expert.  This choses the unlabeled
        examples with the highest entropy.
        """
        number_to_select: int = self.parameters["number_to_select_per_query"]
        number_of_feature_vectors: int = (
            self.dataset_handler.get_all_feature_vectors().shape[0]
        )

        # Make sure the pool to select from is large enough
        excluded_indices: Set = set(labeled_indices) | set(validation_indices)
        if number_to_select + len(excluded_indices) > number_of_feature_vectors:
            raise ValueError(
                f"Cannot not select {number_to_select} unlabeled feature vectors; only"
                f" {number_of_feature_vectors - len(excluded_indices)} remain."
            )

        predictions: NDArray = self.predictions
        if np.amax(predictions) <= 0.0:
            # Convert log_softmax to softmax
            predictions = np.exp(predictions)
        # We assume that via "softmax" or similar, the values are already non-negative.
        # Probably they also already sum to 1.0, but let's force that anyway.
        predictions = predictions / predictions.sum(axis=-1, keepdims=True)
        # Scipy will (normalize row values to sum to 1 and then) compute the entropy of
        # each row.  We negate the entropy so that the distributions that are near
        # uniform have the smallest (i.e., most negative) values.
        predict_score = -scipy.stats.entropy(predictions, axis=-1)
        # Make the currently labeled examples look confident, so that they won't be
        # selected
        if len(excluded_indices):
            predict_score[np.fromiter(excluded_indices, dtype=np.int64)] = 2
        # Find the lowest scoring examples
        predict_order: NDArray = np.argsort(predict_score)[0:number_to_select]
        return predict_order


class BaldStrategyHandler(GenericStrategyHandler):
    def __init__(self) -> None:
        super(BaldStrategyHandler, self).__init__()

    def select_next_indices(
        self,
        labeled_indices: NDArray,
        validation_indices: NDArray = np.array((), dtype=np.int64),
    ) -> NDArray:
        """
        Select new examples to be labeled by the expert.  This choses the unlabeled
        examples based upon the BALD criterion.  (See also BatchBaldStrategyHandler.)
        """
        print(f"self.predictions.shape = {self.predictions.shape}")
        raise NotImplementedError(
            "BaldStrategyHandler::select_next_indices is not yet implemented."
        )


class BatchBaldStrategyHandler(GenericStrategyHandler):
    def __init__(self) -> None:
        super(BatchBaldStrategyHandler, self).__init__()

    def select_next_indices(
        self,
        labeled_indices: NDArray,
        validation_indices: NDArray = np.array((), dtype=np.int64),
    ) -> NDArray:
        """
        Select new examples to be labeled by the expert.  This choses the unlabeled
        examples based upon the Batch-BALD criterion.  (See also BaldStrategyHandler.)
        """
        import torch
        import batchbald_redux as bbald
        import batchbald_redux.batchbald

        # !!! Check that we have a torch model, not tensorflow
        number_to_select: int = self.parameters["number_to_select_per_query"]
        # Use the subset of self.predictions that excludes labeled_indices and
        # validation_indices
        available_indices = np.fromiter(
            set(range(self.predictions.shape[0]))
            - (set(labeled_indices) | set(validation_indices)),
            dtype=np.int64,
        )

        # Check that there are enough available indices left
        if number_to_select > available_indices.shape[0]:
            raise ValueError(
                f"Cannot not select {number_to_select} unlabeled feature vectors; only"
                f" {available_indices.shape[0]} remain."
            )

        num_samples: int = 100000
        with torch.no_grad():
            candidates = bbald.batchbald.get_batchbald_batch(
                torch.from_numpy(self.predictions[available_indices]),
                number_to_select,
                num_samples,
                dtype=torch.double,
            )
        if False:
            print(f"type(candidates) = {type(candidates)}")
            print(f"dir(candidates) = {dir(candidates)}")
            print(f"type(candidates.indices) = {type(candidates.indices)}")
            print(f"type(candidates.scores) = {type(candidates.scores)}")
            print(f"len(candidates.indices) = {len(candidates.indices)}")
            print(f"len(candidates.scores) = {len(candidates.scores)}")
            print(f"type(candidates.indices[0]) = {type(candidates.indices[0])}")
            print(f"type(candidates.scores[0]) = {type(candidates.scores[0])}")
            print(f"candidates.indices = {candidates.indices}")
            print(f"candidates.scores = {candidates.scores}")
        if False:
            raise NotImplementedError(
                "BatchBaldStrategyHandler::select_next_indices is not yet implemented."
            )
        return available_indices[candidates.indices]
