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


def test_imports():
    """Purpose: Test that needed packages are available"""

    import al_bench as alb
    import h5py as h5
    import numpy as np
    import tensorflow as tf


def test_high_level():
    """Purpose: Test that high-level operations work"""
    import al_bench as alb

    ## Create the three top-level handlers and the main tool
    my_dataset_handler = alb.dataset.BasicDatasetHandler()
    my_model_handler = alb.model.TensorFlowModelHandler()
    my_strategy_handler = alb.strategy.ToyStrategyHandler()

    # Specify some testing parameters
    parameters = dict(
        number_of_superpixels=1000,
        number_of_features=2048,
        number_of_categories_by_label=[5, 7],
        label_to_test=0,
    )

    ## Try trivial exercises on the handler interfaces
    exercise_database_handler(my_dataset_handler, **parameters)
    exercise_model_handler(my_model_handler, **parameters)
    exercise_strategy_handler(my_strategy_handler, **parameters)


def test_active_learning():
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

    # Specify some testing parameters
    parameters = dict(
        number_of_superpixels=100,
        number_of_features=64,
        number_of_categories_by_label=[3],
        label_to_test=0,
    )

    # Create some inputs
    my_features, my_label_definitions, my_labels = create_dataset(**parameters)
    tensorflow_model = create_tensorflow_model(**parameters)

    ## Create the three top-level handlers and the main tool
    my_dataset_handler = alb.dataset.BasicDatasetHandler()
    my_model_handler = alb.model.TensorFlowModelHandler()
    my_strategy_handler = alb.strategy.ToyStrategyHandler()

    # Tell components about each other
    my_dataset_handler.set_all_features(my_features)
    my_dataset_handler.set_all_labels(my_labels)
    my_dataset_handler.set_all_label_definitions(my_label_definitions)
    my_model_handler.set_model(tensorflow_model)
    my_strategy_handler.set_dataset_handler(my_dataset_handler)
    my_strategy_handler.set_model_handler(my_model_handler)

    number_iterations = 5
    my_strategy_handler.set_learning_parameters(
        maximum_iterations=number_iterations,
        label_of_interest=parameters["label_to_test"],
        number_to_select_per_iteration=int(
            parameters["number_of_superpixels"] // (number_iterations + 1)
        ),
    )

    # Start with nothing labeled yet
    currently_labeled_examples = set()
    my_strategy_handler.run(currently_labeled_examples)


"""
Create a toy set of features
"""


def create_dataset(
    number_of_superpixels,
    number_of_features,
    number_of_categories_by_label,
    **kwargs,
):
    import numpy as np

    my_features = np.random.normal(
        0, 1, size=(number_of_superpixels, number_of_features)
    )

    if not isinstance(number_of_categories_by_label, (list, tuple)):
        number_of_categories_by_label = [number_of_categories_by_label]

    ## Note that apparently TensorFlow requires that the labels be consecutive integers
    ## starting with zero.  So, we will use -1 for "unknown".
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
        np.array(np.clip(np.floor(my_features[:, 0:1] ** 2 * count + 1), 0, count - 1), dtype=int)
        for count in number_of_categories_by_label
    ]
    if len(my_labels) == 1:
        my_labels = my_labels[0]
    else:
        my_labels = np.append(*my_labels, 1)

    return my_features, my_label_definitions, my_labels


"""
Create a toy TensorFlow model that has the right shape for inputs and outputs
"""


def create_tensorflow_model(
    number_of_features, number_of_categories_by_label, label_to_test, **kwargs
):
    import tensorflow as tf

    number_of_categories = number_of_categories_by_label[label_to_test]
    model = tf.keras.models.Sequential(
        [
            tf.keras.Input(shape=(number_of_features,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(number_of_categories),
        ],
        name=f"{number_of_categories}_labels_from_{number_of_features}_features",
    )
    return model


def exercise_database_handler(
    my_dataset_handler,
    number_of_superpixels,
    number_of_features,
    number_of_categories_by_label,
    label_to_test,
    **kwargs,
):
    # Write me!!! to test read_all_features_from_h5py(self, filename, data_name="features"):
    # Write me!!! to test write_all_features_to_h5py(self, filename, data_name="features"):
    # Write me!!! to test set_some_features(self, feature_indices, features):
    # Write me!!! to test get_some_features(self, feature_indices):
    # Write me!!! to test read_all_labels_from_h5py(self, filename, data_name="labels"):
    # Write me!!! to test write_all_labels_to_h5py(self, filename, data_name="labels"):
    # Write me!!! to test set_some_labels(self, label_indices, labels):
    # Write me!!! to test get_some_labels(self, label_indices):
    # Write me!!! to test set_all_dictionaries(self, dictionaries):
    # Write me!!! to test get_all_dictionaries(self):
    # Write me!!! to test clear_all_dictionaries(self):
    # Write me!!! to test set_some_dictionaries(self, dictionary_indices, dictionaries):
    # Write me!!! to test get_some_dictionaries(self, dictionary_indices):
    # Write me!!! to test get_all_label_definitions(self):

    # raise NotImplementedError("Not implemented")
    assert my_dataset_handler.get_all_features() is None
    assert my_dataset_handler.get_all_labels() is None

    # Create random feature vectors
    my_features, my_label_definitions, my_labels = create_dataset(
        number_of_superpixels, number_of_features, number_of_categories_by_label
    )

    my_dataset_handler.set_all_features(my_features)
    assert my_dataset_handler.get_all_features() is my_features
    my_dataset_handler.clear_all_features()
    assert my_dataset_handler.get_all_features() is None

    my_dataset_handler.set_all_labels(my_labels)
    assert my_dataset_handler.get_all_labels() is my_labels
    my_dataset_handler.set_all_labels(my_labels[label_to_test])
    assert (my_dataset_handler.get_all_labels() == my_labels[label_to_test]).all()
    my_dataset_handler.clear_all_labels()
    assert my_dataset_handler.get_all_labels() is None

    my_dataset_handler.set_all_features(my_features)
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
    # Write me!!! to test set_dataset_handler(self, dataset_handler):
    # Write me!!! to test set_training_parameters(self):
    # Write me!!! to test train(self):
    # Write me!!! to test predict(self):

    tensorflow_model = create_tensorflow_model(
        number_of_features, number_of_categories_by_label, label_to_test
    )
    my_model_handler.set_model(tensorflow_model)
    pass


def exercise_strategy_handler(my_strategy_handler, **kwargs):
    # Write me!!! to test set_dataset_handler(self, dataset_handler):
    # Write me!!! to test get_dataset_handler(self):
    # Write me!!! to test set_model_handler(self, model_handler):
    # Write me!!! to test get_model_handler(self):
    # Write me!!! to test set_desired_outputs(self):
    # Write me!!! to test set_scoring_metric(self, scoring_metric):
    # Write me!!! to test get_scoring_metric(self):
    # Write me!!! to test clear_scoring_metric(self):
    # Write me!!! to test set_diversity_metric(self, diversity_metric):
    # Write me!!! to test get_diversity_metric(self):
    # Write me!!! to test clear_diversity_metric(self):
    # Write me!!! to test select_next_examples(self, currently_labeled_examples):

    pass


if __name__ == "__main__":
    test_imports()
    test_high_level()
    test_active_learning()
