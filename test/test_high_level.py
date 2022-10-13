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


def test_imports():
    """Purpose: Test that needed packages are available"""
    import al_bench as alb
    import h5py as h5
    import numpy as np
    import os
    import re
    import tensorflow as tf
    import torch

    pass


def test_dataset_handler_interface():
    """Purpose: Test that high-level operations work"""
    import al_bench as alb

    # Specify some testing parameters
    parameters = dict(
        number_of_superpixels=1000,
        number_of_features=2048,
        number_of_categories_by_label=[5, 7],
        label_to_test=0,
    )

    # Try trivial exercises on the handler interface
    for DatasetHandler in (alb.dataset.GenericDatasetHandler,):
        my_dataset_handler = DatasetHandler()
        exercise_dataset_handler(my_dataset_handler, **parameters)


def test_model_handler_interface():
    """Purpose: Test that high-level operations work"""
    import al_bench as alb

    # Specify some testing parameters
    parameters = dict(
        number_of_superpixels=1000,
        number_of_features=2048,
        number_of_categories_by_label=[5, 7],
        label_to_test=0,
    )

    # Try trivial exercises on the handler interface
    for ModelHandler in (
        alb.model.TensorFlowModelHandler,
        alb.model.PyTorchModelHandler,
    ):
        my_model_handler = ModelHandler()
        exercise_model_handler(my_model_handler, **parameters)


def test_strategy_handler_interface():
    """Purpose: Test that high-level operations work"""
    import al_bench as alb

    # Specify some testing parameters
    parameters = dict(
        number_of_superpixels=1000,
        number_of_features=2048,
        number_of_categories_by_label=[5, 7],
        label_to_test=0,
    )

    # Try trivial exercises on the handler interface
    for StrategyHandler in (
        alb.strategy.RandomStrategyHandler,
        alb.strategy.LeastConfidenceStrategyHandler,
        alb.strategy.LeastMarginStrategyHandler,
        alb.strategy.EntropyStrategyHandler,
    ):
        my_strategy_handler = StrategyHandler()
        exercise_strategy_handler(my_strategy_handler, **parameters)


def deeply_numeric(x):
    import numpy as np

    return (
        isinstance(
            x,
            (
                int,
                float,
                np.float32,
            ),
        )
        or (isinstance(x, np.ndarray) and len(x.shape) == 0 and deeply_numeric(x[()]))
        or all([deeply_numeric(e) for e in x])
    )


def test_handler_combinations():
    """
    Keys supported by the `parameters` dict
    ---------------------------------------
    number_of_superpixels: int
        For example 10000 superpixels in our dataset to learn from and predict for.
    number_of_features: int
        For example, 64 or 2048 floats per superpixel to describe it.
    number_of_categories_by_label: list of int
        Often there is only one label per feature vector but we support multiple labels.
        For a feature vector, each label can either be "Unknown" or one of the specified
        number of known categories.  With multiple labels, use, e.g., [5,7] when one
        label has 5 categories and a second label has 7 categories:
    label_to_test: int
        The index into number_of_categories_by_label and into my_labels that specifies
        the label that we will test.
    """

    import al_bench as alb
    import datetime
    import os
    import re

    # Specify some testing parameters
    parameters = dict(
        number_of_superpixels=4598,
        number_of_features=1280,
        number_of_categories_by_label=[4],
        label_to_test=0,
    )
    number_iterations = 20
    number_per_iteration = 5

    combination_index = 0
    for dataset_creator, DatasetHandler in (
        (
            create_dataset,
            alb.dataset.GenericDatasetHandler,
        ),
        (
            create_dataset_4598_1280_4,
            alb.dataset.GenericDatasetHandler,
        ),
    ):
        for model_creator, ModelHandler in (
            (
                create_toy_tensorflow_model,
                alb.model.TensorFlowModelHandler,
            ),
            (
                create_toy_pytorch_model,
                alb.model.PyTorchModelHandler,
            ),
            (
                create_tensorflow_model_with_dropout,
                alb.model.TensorFlowModelHandler,
            ),
            (
                create_pytorch_model_with_dropout,
                alb.model.PyTorchModelHandler,
            ),
        ):
            for StrategyHandler in (
                alb.strategy.RandomStrategyHandler,
                alb.strategy.LeastConfidenceStrategyHandler,
                alb.strategy.LeastMarginStrategyHandler,
                alb.strategy.EntropyStrategyHandler,
            ):
                # Create fresh handlers and components
                my_feature_vectors, my_label_definitions, my_labels = dataset_creator(
                    **parameters
                )
                my_dataset_handler = DatasetHandler()
                my_dataset_handler.set_all_feature_vectors(my_feature_vectors)
                my_dataset_handler.set_all_label_definitions(my_label_definitions)
                my_dataset_handler.set_all_labels(my_labels)
                my_dataset_handler.set_validation_indices(
                    list(range(my_feature_vectors.shape[0] // 10))
                )

                my_model = model_creator(**parameters)
                my_model_handler = ModelHandler()
                my_model_handler.set_model(my_model)

                my_strategy_handler = StrategyHandler()
                my_strategy_handler.set_dataset_handler(my_dataset_handler)
                my_strategy_handler.set_model_handler(my_model_handler)
                my_strategy_handler.set_learning_parameters(
                    maximum_iterations=number_iterations,
                    label_of_interest=parameters["label_to_test"],
                    number_to_select_per_iteration=number_per_iteration,
                )

                # Start with nothing labeled yet
                currently_labeled_examples = set()

                # Go!
                combination_name = "-".join(
                    [
                        re.search(
                            r"<class 'al_bench\.dataset\.(.*)'>",
                            f"{type(my_dataset_handler)}",
                        ).group(1),
                        re.search(
                            r"<class 'al_bench\.model\.(.*)'>",
                            f"{type(my_model_handler)}",
                        ).group(1),
                        re.search(
                            r"<class 'al_bench\.strategy\.(.*)'>",
                            f"{type(my_strategy_handler)}",
                        ).group(1),
                        datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S.%f"),
                        f"{combination_index:06d}",
                    ]
                )
                print(f"Exercise combination: {combination_name}")
                combination_index += 1
                my_strategy_handler.run(currently_labeled_examples)
                my_log = my_strategy_handler.get_log()
                assert (
                    my_log is not None
                    and isinstance(my_log, list)
                    and all([isinstance(e, dict) for e in my_log])
                    and all(
                        [isinstance(e["utcnow"], datetime.datetime) for e in my_log]
                    )
                    and all(
                        [
                            isinstance(e["model_step"], alb.model.ModelStep)
                            for e in my_log
                        ]
                    )
                    and all(
                        [
                            isinstance(e["logs"], dict)
                            for e in my_log
                            if e["logs"] is not None
                        ]
                    )
                    and all(
                        [
                            deeply_numeric(v)
                            for e in my_log
                            if e["logs"] is not None
                            for v in e["logs"].values()
                        ]
                    )
                )
                my_strategy_handler.write_train_log_to_tensorboard_file(
                    log_dir=os.path.join("runs", combination_name)
                )


def create_dataset(
    number_of_superpixels,
    number_of_features,
    number_of_categories_by_label,
    **kwargs,
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
    number_of_superpixels,
    number_of_features,
    number_of_categories_by_label,
    **kwargs,
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


def exercise_dataset_handler(
    my_dataset_handler,
    number_of_superpixels,
    number_of_features,
    number_of_categories_by_label,
    label_to_test,
    **kwargs,
):
    print(
        f"exercise_dataset_handler on type(my_dataset_handler) = "
        f"{type(my_dataset_handler)}"
    )
    # !!! Test read_all_feature_vectors_from_h5py(self, filename, data_name="features"):
    # !!! Test write_all_feature_vectors_to_h5py(self, filename, data_name="features"):
    # !!! Test set_some_feature_vectors(self, feature_vector_indices, feature_vectors):
    # !!! Test get_some_feature_vectors(self, feature_vector_indices):
    # !!! Test read_all_labels_from_h5py(self, filename, data_name="labels"):
    # !!! Test write_all_labels_to_h5py(self, filename, data_name="labels"):
    # !!! Test set_some_labels(self, label_indices, labels):
    # !!! Test get_some_labels(self, label_indices):
    # !!! Test set_all_dictionaries(self, dictionaries):
    # !!! Test get_all_dictionaries(self):
    # !!! Test clear_all_dictionaries(self):
    # !!! Test set_some_dictionaries(self, dictionary_indices, dictionaries):
    # !!! Test get_some_dictionaries(self, dictionary_indices):
    # !!! Test get_all_label_definitions(self):
    # !!! Test set_validation_indices(self, validation_indices):
    # !!! Test get_validation_indices(self):
    # !!! Test clear_validation_indices(self):

    # raise NotImplementedError("Not implemented")
    assert my_dataset_handler.get_all_feature_vectors() is None
    assert my_dataset_handler.get_all_labels() is None

    # Create random feature vectors
    my_feature_vectors, my_label_definitions, my_labels = create_dataset(
        number_of_superpixels, number_of_features, number_of_categories_by_label
    )

    my_dataset_handler.set_all_feature_vectors(my_feature_vectors)
    assert my_dataset_handler.get_all_feature_vectors() is my_feature_vectors
    my_dataset_handler.clear_all_feature_vectors()
    assert my_dataset_handler.get_all_feature_vectors() is None

    my_dataset_handler.set_all_labels(my_labels)
    assert my_dataset_handler.get_all_labels() is my_labels
    my_dataset_handler.set_all_labels(my_labels[label_to_test])
    assert (my_dataset_handler.get_all_labels() == my_labels[label_to_test]).all()
    my_dataset_handler.clear_all_labels()
    assert my_dataset_handler.get_all_labels() is None

    my_dataset_handler.set_all_feature_vectors(my_feature_vectors)
    my_dataset_handler.set_all_labels(my_labels)
    my_dataset_handler.set_all_label_definitions(my_label_definitions)
    my_dataset_handler.check_data_consistency()


def exercise_model_handler(
    my_model_handler,
    number_of_features,
    number_of_categories_by_label,
    label_to_test,
    **kwargs,
):
    import al_bench as alb

    print(
        f"exercise_model_handler on type(my_model_handler) = "
        f"{type(my_model_handler)}"
    )
    model = None
    if isinstance(my_model_handler, alb.model.TensorFlowModelHandler):
        model = create_toy_tensorflow_model(
            number_of_features, number_of_categories_by_label, label_to_test
        )
    if isinstance(my_model_handler, alb.model.PyTorchModelHandler):
        model = create_toy_pytorch_model(
            number_of_features, number_of_categories_by_label, label_to_test
        )
    my_model_handler.set_model(model)

    # !!! Test set_dataset_handler(self, dataset_handler):
    # !!! Test set_training_parameters(self):
    # !!! Test train(self):
    # !!! Test predict(self):


def exercise_strategy_handler(my_strategy_handler, **kwargs):
    print(
        f"exercise_strategy_handler on type(my_strategy_handler) = "
        f"{type(my_strategy_handler)}"
    )
    # !!! Test set_dataset_handler(self, dataset_handler):
    # !!! Test get_dataset_handler(self):
    # !!! Test set_model_handler(self, model_handler):
    # !!! Test get_model_handler(self):
    # !!! Test set_desired_outputs(self):
    # !!! Test set_scoring_metric(self, scoring_metric):
    # !!! Test get_scoring_metric(self):
    # !!! Test clear_scoring_metric(self):
    # !!! Test set_diversity_metric(self, diversity_metric):
    # !!! Test get_diversity_metric(self):
    # !!! Test clear_diversity_metric(self):
    # !!! Test select_next_examples(self, currently_labeled_examples):
    pass


if __name__ == "__main__":
    test_imports()
    test_dataset_handler_interface()
    test_model_handler_interface()
    test_strategy_handler_interface()
    test_handler_combinations()
