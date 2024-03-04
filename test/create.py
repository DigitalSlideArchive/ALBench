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

from typing import List, Mapping, Optional, Tuple

import numpy as np

from check import NDArrayFloat, NDArrayInt, SequenceFloat, SequenceInt


def create_dataset(
    number_of_superpixels: int,
    number_of_features: int,
    number_of_categories_by_label: List[int],
    **kwargs,
) -> Tuple[NDArrayFloat, List[Mapping], NDArrayInt]:
    """
    Create a toy set of feature vectors
    """
    rng = np.random.default_rng()
    my_feature_vectors: NDArrayFloat
    my_feature_vectors = rng.normal(
        0, 1, size=(number_of_superpixels, number_of_features)
    ).astype(np.float64)

    # Note that apparently TensorFlow requires that the labels be consecutive integers
    # starting with zero.  So, we will use -1 for "unknown".
    my_label_definitions: List[Mapping]
    my_label_definitions = [
        {
            -1: {"description": f"Label{label_index}Unknown"},
            **{
                category_index: {
                    "description": f"Label{label_index}Category{category_index}"
                }
                for category_index in range(number_of_categories)
            },
        }
        for label_index, number_of_categories in enumerate(
            number_of_categories_by_label
        )
    ]

    # Create a random label for each superpixel.  Avoid -1, which we are using for
    # "unknown".
    my_labels: NDArrayInt
    my_labels = np.concatenate(
        [
            np.array(
                np.clip(
                    np.floor(my_feature_vectors[:, 0:1] ** 2 * count + 1), 0, count - 1
                ),
                dtype=int,
            )
            for count in number_of_categories_by_label
        ],
        axis=1,
    )

    return my_feature_vectors, my_label_definitions, my_labels


def create_dataset_4598_1280_4(
    number_of_superpixels: int,
    number_of_features: int,
    number_of_categories_by_label,
    **kwargs,
) -> Tuple[NDArrayFloat, List[Mapping], NDArrayInt]:
    import h5py as h5

    """Use the dataset from test/TCGA-A2-A0D0-DX1_xmin68482_ymin39071_MPP-0.2500.h5py"""
    filename: str = "TCGA-A2-A0D0-DX1_xmin68482_ymin39071_MPP-0.2500.h5py"
    with h5.File(filename) as ds:
        my_feature_vectors: NDArrayFloat = np.array(ds["features"])
        my_labels: NDArrayInt = np.array(ds["labels"])
    my_label_definitions: List[Mapping]
    my_label_definitions = [
        {
            0: {"description": "other"},
            1: {"description": "tumor"},
            2: {"description": "stroma"},
            3: {"description": "infiltrate"},
        }
    ]
    assert number_of_superpixels == my_feature_vectors.shape[0]
    assert number_of_features == my_feature_vectors.shape[1]
    assert isinstance(number_of_categories_by_label, (list, tuple))
    assert len(number_of_categories_by_label) == len(my_label_definitions)
    assert number_of_categories_by_label[0] == len(my_label_definitions[0])

    return my_feature_vectors, my_label_definitions, my_labels


def create_toy_tensorflow_model(
    number_of_features: int,
    number_of_categories_by_label: List[int],
    label_to_test: int,
    hidden_units: int = 128,
    **kwargs,
):
    """
    Create a toy TensorFlow model that has the right shape for inputs and outputs
    """
    import tensorflow as tf

    number_of_categories: int = number_of_categories_by_label[label_to_test]
    model = tf.keras.models.Sequential(
        [
            tf.keras.Input(shape=(number_of_features,)),
            tf.keras.layers.Dense(hidden_units, activation="relu"),
            tf.keras.layers.Dense(number_of_categories, activation="softmax"),
        ],
        name=f"{number_of_categories}_labels_from_{number_of_features}_features",
    )
    return model


def create_tensorflow_model_with_dropout(
    number_of_features: int,
    number_of_categories_by_label: List[int],
    label_to_test: int,
    hidden_units: int = 32,
    dropout: float = 0.3,
    noise_shape: Optional[SequenceInt] = None,
    seed: int = 145,
    **kwargs,
):
    """
    Create a toy TensorFlow model that has the right shape for inputs and outputs
    """
    import tensorflow as tf

    number_of_categories: int = number_of_categories_by_label[label_to_test]
    model = tf.keras.models.Sequential(
        [
            tf.keras.Input(shape=(number_of_features,)),
            tf.keras.layers.Dense(hidden_units, activation="relu"),
            tf.keras.layers.Dropout(dropout, noise_shape=noise_shape, seed=seed),
            tf.keras.layers.Dense(number_of_categories, activation="softmax"),
        ],
        name=(
            f"{number_of_categories}_labels_from_{number_of_features}_features_with_"
            f"dropout_{dropout}"
        ),
    )
    return model


def create_toy_pytorch_model(
    number_of_features: int,
    number_of_categories_by_label: List[int],
    label_to_test: int,
    hidden_units: int = 128,
    **kwargs,
):
    """
    Create a toy PyTorch model that has the right shape for inputs and outputs
    """
    import torch

    class TorchToy(torch.nn.modules.module.Module):
        def __init__(self, number_of_features: int, number_of_categories: int):
            super(TorchToy, self).__init__()
            self.fc1 = torch.nn.Linear(number_of_features, hidden_units)
            self.relu1 = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(hidden_units, number_of_categories)
            self.softmax1 = torch.nn.Softmax(dim=-1)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu1(x)
            x = self.fc2(x)
            x = self.softmax1(x)
            return x

    number_of_categories: int = number_of_categories_by_label[label_to_test]
    model = TorchToy(number_of_features, number_of_categories)
    return model


def create_pytorch_model_with_dropout(
    number_of_features: int,
    number_of_categories_by_label: List[int],
    label_to_test: int,
    hidden_units: int = 32,
    dropout: float = 0.3,
    noise_shape: Optional[SequenceInt] = None,
    seed: int = 145,
    **kwargs,
):
    """
    Create a toy PyTorch model that has the right shape for inputs and outputs
    """
    import torch

    class TorchToyWithDropout(torch.nn.modules.module.Module):
        def __init__(self, number_of_features: int, number_of_categories: int):
            super(TorchToyWithDropout, self).__init__()
            self.fc1 = torch.nn.Linear(number_of_features, hidden_units)
            self.relu1 = torch.nn.ReLU()
            self.dropout1 = torch.nn.Dropout(p=dropout)
            self.fc2 = torch.nn.Linear(hidden_units, number_of_categories)
            self.softmax1 = torch.nn.Softmax(dim=-1)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu1(x)
            x = self.dropout1(x)
            x = self.fc2(x)
            x = self.softmax1(x)
            return x

    number_of_categories: int = number_of_categories_by_label[label_to_test]
    model = TorchToyWithDropout(number_of_features, number_of_categories)
    return model


def rng_gamma(
    alpha: float,
    beta: float = 1.0,
    shape: SequenceInt = (),
    rng: np.random._generator.Generator = np.random.default_rng(),
) -> NDArrayFloat:
    num_factors: int = 100
    response: NDArrayFloat = np.zeros(shape=shape, dtype=float)
    linearized: NDArrayFloat = np.reshape(response, (-1,))
    for i in range(linearized.size):
        random_values: NDArrayFloat = rng.random((num_factors,))
        value: float = alpha
        for k in range(num_factors):
            ratio = 1.0 / (alpha + k)
            value *= (1.0 + ratio) * random_values[k] ** ratio
        linearized[i] = value + 1e-12
    return response / beta


def rng_dirichlet(
    alpha: SequenceFloat,
    shape: SequenceInt,
    rng: np.random._generator.Generator = np.random.default_rng(),
) -> NDArrayFloat:
    dimensions: int = len(alpha)
    # Note that returned shape is the concatenation of shape + (dimensions,)
    response: NDArrayFloat = np.zeros(tuple(shape) + (dimensions,))
    for i in range(dimensions):
        response[..., i] = rng_gamma(alpha=alpha[i], beta=1.0, shape=shape, rng=rng)
    response = response / np.sum(response, axis=-1, keepdims=True)
    return response


def create_dirichlet_predictions(
    num_samples: int,
    num_repeats: int,
    num_classes: int,
    alpha_hyperprior: float = 2.0,
    beta_hyperprior: float = 1.0,
    rng: np.random._generator.Generator = np.random.default_rng(),
) -> Tuple[NDArrayFloat, NDArrayFloat]:
    # Make predictions array where each row sums to 1.0.  Using alpha_hyperprior and
    # beta_hyperprior to parameterize a gamma distribution we draw pseudocounts for
    # every (sample, class) combination.  For a given sample, the (sample, :) values
    # will be the pseudocounts for the Dirichlet distribution from which the (repeated)
    # predictions for the sample will be drawn.  In particular a sample for which the
    # pseudocounts are all large will produce (Bayesian) predictions that are similar; a
    # sample for which the pseudocounts are all small will produce (Bayesian)
    # predictions that have similar mean but are not significantly similar.
    remove_repeats = num_repeats is None
    if remove_repeats:
        num_repeats = 1

    predictions: NDArrayFloat = np.zeros((num_samples, num_repeats, num_classes))
    sample_pseudocounts: NDArrayFloat
    sample_pseudocounts = rng_gamma(
        alpha=alpha_hyperprior,
        beta=beta_hyperprior,
        shape=(num_samples, num_classes),
        rng=rng,
    )
    for s in range(num_samples):
        predictions[s, :, :] = rng_dirichlet(
            alpha=sample_pseudocounts[s, :], shape=(num_repeats,), rng=rng
        )
    if remove_repeats:
        predictions = np.reshape(predictions, (num_samples, num_classes))
    return sample_pseudocounts, predictions


def create_predictions(
    num_samples: int,
    num_repeats: int,
    num_classes: int,
    rng: np.random._generator.Generator = np.random.default_rng(),
) -> NDArrayFloat:
    predictions: NDArrayFloat
    sample_pseudocounts: SequenceFloat = (1.0,) * num_classes
    if num_repeats is None:
        predictions = rng_dirichlet(
            alpha=sample_pseudocounts, shape=(num_samples,), rng=rng
        )
    else:
        predictions = rng_dirichlet(
            alpha=sample_pseudocounts, shape=(num_samples, num_repeats), rng=rng
        )
    return predictions
