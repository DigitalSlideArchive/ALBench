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


def create_dataset(
    number_of_superpixels, number_of_features, number_of_categories_by_label, **kwargs
):
    """
    Create a toy set of feature vectors
    """
    import numpy as np

    rng = np.random.default_rng()
    my_feature_vectors = rng.normal(
        0, 1, size=(number_of_superpixels, number_of_features)
    ).astype(np.float32)

    if not isinstance(number_of_categories_by_label, (list, tuple)):
        number_of_categories_by_label = [number_of_categories_by_label]

    # Note that apparently TensorFlow requires that the labels be consecutive integers
    # starting with zero.  So, we will use -1 for "unknown".
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
    my_labels = [
        np.array(
            np.clip(
                np.floor(my_feature_vectors[:, 0:1] ** 2 * count + 1), 0, count - 1
            ),
            dtype=int,
        )
        for count in number_of_categories_by_label
    ]
    if len(my_labels) == 1:
        my_labels = my_labels[0]
    else:
        my_labels = np.append(*my_labels, 1)

    return my_feature_vectors, my_label_definitions, my_labels


def create_dataset_4598_1280_4(
    number_of_superpixels, number_of_features, number_of_categories_by_label, **kwargs
):
    import h5py as h5
    import numpy as np

    """Use the dataset from test/TCGA-A2-A0D0-DX1_xmin68482_ymin39071_MPP-0.2500.h5py"""
    filename = "TCGA-A2-A0D0-DX1_xmin68482_ymin39071_MPP-0.2500.h5py"
    with h5.File(filename) as ds:
        my_feature_vectors = np.array(ds["features"])
        my_labels = np.array(ds["labels"])
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
    assert isinstance(number_of_categories_by_label, list)
    assert len(number_of_categories_by_label) == len(my_label_definitions)
    assert number_of_categories_by_label[0] == len(my_label_definitions[0])

    return my_feature_vectors, my_label_definitions, my_labels


def create_toy_tensorflow_model(
    number_of_features,
    number_of_categories_by_label,
    label_to_test,
    hidden_units=128,
    **kwargs,
):
    """
    Create a toy TensorFlow model that has the right shape for inputs and outputs
    """
    import tensorflow as tf

    number_of_categories = number_of_categories_by_label[label_to_test]
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
    number_of_features,
    number_of_categories_by_label,
    label_to_test,
    hidden_units=32,
    dropout=0.3,
    noise_shape=None,
    seed=145,
    **kwargs,
):
    """
    Create a toy TensorFlow model that has the right shape for inputs and outputs
    """
    import tensorflow as tf

    number_of_categories = number_of_categories_by_label[label_to_test]
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
    number_of_features,
    number_of_categories_by_label,
    label_to_test,
    hidden_units=128,
    **kwargs,
):
    """
    Create a toy PyTorch model that has the right shape for inputs and outputs
    """
    import torch

    class TorchToy(torch.nn.modules.module.Module):
        def __init__(self, number_of_features, number_of_categories):
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

    number_of_categories = number_of_categories_by_label[label_to_test]
    model = TorchToy(number_of_features, number_of_categories)
    return model


def create_pytorch_model_with_dropout(
    number_of_features,
    number_of_categories_by_label,
    label_to_test,
    hidden_units=32,
    dropout=0.3,
    noise_shape=None,
    seed=145,
    **kwargs,
):
    """
    Create a toy PyTorch model that has the right shape for inputs and outputs
    """
    import torch

    class TorchToyWithDropout(torch.nn.modules.module.Module):
        def __init__(self, number_of_features, number_of_categories):
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

    number_of_categories = number_of_categories_by_label[label_to_test]
    model = TorchToyWithDropout(number_of_features, number_of_categories)
    return model
