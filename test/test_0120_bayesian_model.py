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
import al_bench as alb
import al_bench.strategy
import batchbald_redux as bbald
import batchbald_redux.consistent_mc_dropout
import batchbald_redux.repeated_mnist
import numpy as np
import os
import random
import shutil
import torch
from check import NDArrayFloat, NDArrayInt
from typing import Mapping, List, Tuple


class BayesianCNN(bbald.consistent_mc_dropout.BayesianModule):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        self.conv1: torch.Module = torch.nn.Conv2d(1, 32, kernel_size=5)
        self.conv1_drop: torch.Module
        self.conv1_drop = bbald.consistent_mc_dropout.ConsistentMCDropout2d()
        self.conv2: torch.Module = torch.nn.Conv2d(32, 64, kernel_size=5)
        self.conv2_drop: torch.Module
        self.conv2_drop = bbald.consistent_mc_dropout.ConsistentMCDropout2d()
        self.fc1: torch.Module = torch.nn.Linear(1024, 128)
        self.fc1_drop: torch.Module = bbald.consistent_mc_dropout.ConsistentMCDropout()
        self.fc2: torch.Module = torch.nn.Linear(128, num_classes)

    def mc_forward_impl(self, input: torch.Tensor) -> torch.Tensor:
        input = self.conv1(input)
        input = self.conv1_drop(input)
        input = torch.nn.functional.max_pool2d(input, 2)
        input = torch.nn.functional.relu(input)
        input = self.conv2(input)
        input = self.conv2_drop(input)
        input = torch.nn.functional.max_pool2d(input, 2)
        input = torch.nn.functional.relu(input)
        input = input.view(-1, 1024)
        input = self.fc1(input)
        input = self.fc1_drop(input)
        input = torch.nn.functional.relu(input)
        input = self.fc2(input)
        input = torch.nn.functional.log_softmax(input, dim=1)

        return input


def test_0120_bayesian_model() -> None:
    mnist: Tuple[torch.Dataset, torch.Dataset]
    mnist = bbald.repeated_mnist.create_repeated_MNIST_dataset(
        num_repetitions=1, add_noise=False
    )
    train_dataset: torch.Dataset
    test_dataset: torch.Dataset
    train_dataset, test_dataset = mnist

    # al_bench datasets are supplied as numpy arrays.
    # Build the numpy arrays from a subset of the data
    train_dataset_list: List[Tuple[torch.Tensor, int]]
    train_dataset_list = random.sample(list(train_dataset), 500)
    test_dataset_list: List[Tuple[torch.Tensor, int]]
    test_dataset_list = random.sample(list(test_dataset), 50)
    num_training_indices: int = len(train_dataset_list)
    num_validation_indices: int = len(test_dataset_list)
    dataset_list: List[Tuple[torch.Tensor, int]]
    dataset_list = train_dataset_list + test_dataset_list
    # Unzip the data set into separate (unlabeled) input data and their labels.  Data
    # only:
    my_feature_vectors: NDArrayFloat
    my_feature_vectors = np.concatenate([d[0].numpy() for d in dataset_list])

    # Each is list of one label only:
    my_labels: NDArrayInt = np.array([[d[1]] for d in dataset_list])
    # This dataset is the digits "0" through "9" which we will enumerate with the
    # values 0 through 9.
    num_classes: int = 10
    # We have one label per feature_vector so we need a list of one dictionary.
    my_label_definitions: List[Mapping[int, Mapping[str, str]]]
    my_label_definitions = [{i: {"description": repr(i)} for i in range(num_classes)}]
    # We will indicate the validation examples by their indices.
    validation_indices: NDArrayInt
    validation_indices = np.array(
        range(num_training_indices, num_training_indices + num_validation_indices)
    )
    print(f"feature_shape = {my_feature_vectors.shape[1:]}")
    print("Dataset is ready as numpy")

    # Tell al_bench about the dataset
    my_dataset_handler: alb.dataset.AbstractDatasetHandler
    my_dataset_handler = alb.dataset.GenericDatasetHandler()
    my_dataset_handler.set_all_feature_vectors(my_feature_vectors)
    my_dataset_handler.set_all_label_definitions(my_label_definitions)
    my_dataset_handler.set_all_labels(my_labels)
    my_dataset_handler.set_validation_indices(validation_indices)
    print("DatasetHandler is initialized")

    # We'll start the first pass of active learning with some randomly chosen samples
    # from the training data set.
    num_initial_training: int = 20
    currently_labeled_examples: NDArrayInt
    currently_labeled_examples = np.array(
        random.sample(range(num_training_indices), num_initial_training)
    )

    my_pytorch_model: torch.model = BayesianCNN(num_classes)
    print("Created torch model")

    # Tell al_bench about the model
    my_pytorch_model_handler: alb.model.AbstractModelHandler
    my_pytorch_model_handler = alb.model.SamplingBayesianPyTorchModelHandler()
    my_pytorch_model_handler.set_model(my_pytorch_model)
    print("PyTorch model handler built")

    # my_model_handler = my_tensorflow_model_handler
    my_model_handler: alb.model.AbstractModelHandler = my_pytorch_model_handler

    # DELETE OLD LOG FILES
    all_logs_dir = "runs-SamplingBayesian"
    shutil.rmtree(all_logs_dir, ignore_errors=True)

    name: str
    my_strategy_handler: alb.strategy.AbstractStrategyHandler
    for name, my_strategy_handler in (
        ("BatchBALD", alb.strategy.BatchBaldStrategyHandler()),
    ):
        my_strategy_handler.set_dataset_handler(my_dataset_handler)
        my_strategy_handler.set_model_handler(my_model_handler)
        # We've supplied only one label per feature vector, so choose it with
        # label_of_interest=0
        my_strategy_handler.set_learning_parameters(
            label_of_interest=0, maximum_queries=8, number_to_select_per_query=10
        )

        # ################################################################
        # Simulate the strategy.
        my_strategy_handler.run(currently_labeled_examples)
        # ################################################################

        # We will write out collected information to disk.  First say where:
        log_dir: str = os.path.join(all_logs_dir, name)
        # Write accuracy and loss information during training
        my_strategy_handler.write_train_log_for_tensorboard(log_dir=log_dir)
        # Write certainty statistics during active learning
        my_strategy_handler.write_certainty_log_for_tensorboard(log_dir=log_dir)


if __name__ == "__main__":
    test_0120_bayesian_model()
