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

    # Import succeeds
    import al_bench as alb
    import h5py as h5
    import numpy as np
    import tensorflow as tf


def test_high_level():
    """Purpose: Test that high-level operations work"""
    import al_bench as alb

    ## Create handlers
    my_al_benchmark = alb.benchmark.BenchmarkTool()
    my_dataset_handler = alb.dataset.DatasetHandler()
    my_model_handler = alb.model.TensorFlowModelHandler()
    my_strategy_handler = alb.strategy.StrategyHandler()
    assert my_dataset_handler.get_all_features() is None
    assert my_dataset_handler.get_all_labels() is None
    assert my_al_benchmark.get_model_handler() is None
    assert my_al_benchmark.get_dataset_handler() is None

    # Create random feature vectors
    number_of_superpixels = 10
    number_of_features = 6
    number_of_labels, my_features, my_label_definitions, my_labels = create_dataset(
        number_of_superpixels, number_of_features
    )

    ## Exercise the StrategyHandler
    my_al_benchmark.set_model_handler(my_model_handler)
    assert my_al_benchmark.get_model_handler() is my_model_handler
    my_al_benchmark.clear_model_handler()
    assert my_al_benchmark.get_model_handler() is None
    my_al_benchmark.set_model_handler(my_model_handler)
    assert my_al_benchmark.get_model_handler() is my_model_handler

    my_al_benchmark.set_dataset_handler(my_dataset_handler)
    assert my_al_benchmark.get_dataset_handler() is my_dataset_handler
    my_al_benchmark.clear_dataset_handler()
    assert my_al_benchmark.get_dataset_handler() is None
    my_al_benchmark.set_dataset_handler(my_dataset_handler)
    assert my_al_benchmark.get_dataset_handler() is my_dataset_handler

    my_al_benchmark.set_strategy_handler(my_strategy_handler)
    assert my_al_benchmark.get_strategy_handler() is my_strategy_handler
    my_al_benchmark.clear_strategy_handler()
    assert my_al_benchmark.get_strategy_handler() is None
    my_al_benchmark.set_strategy_handler(my_strategy_handler)
    assert my_al_benchmark.get_strategy_handler() is my_strategy_handler

    #!!! my_strategy_handler.set_scoring_metric()
    #!!! my_strategy_handler.set_diversity_metric()

    ## Exercise the DatasetHandler
    my_dataset_handler.set_all_features(my_features)
    assert my_dataset_handler.get_all_features() is my_features
    my_dataset_handler.clear_all_features()
    assert my_dataset_handler.get_all_features() is None
    my_dataset_handler.set_all_features(my_features)
    assert my_dataset_handler.get_all_features() is my_features
    assert my_al_benchmark.get_dataset_handler().get_all_features() is my_features

    #!!! my_dataset_handler.read_all_features_from_h5py()
    #!!! my_dataset_handler.set_some_features()

    my_dataset_handler.set_all_labels(my_labels[:, 0])
    assert (my_dataset_handler.get_all_labels() == my_labels[:, 0]).all()
    my_dataset_handler.clear_all_labels()
    assert my_dataset_handler.get_all_labels() is None
    my_dataset_handler.set_all_labels(my_labels)
    assert my_dataset_handler.get_all_labels() is my_labels

    #!!! my_dataset_handler.read_all_labels_from_h5py()
    #!!! my_dataset_handler.set_some_features()

    #!!! my_dataset_handler.set_all_dictionaries()
    #!!! my_dataset_handler.set_some_dictionaries()

    my_dataset_handler.set_all_label_definitions(my_label_definitions)

    my_dataset_handler.check_data_consistency()

    ## Exercise the ModelHandler
    my_model_handler.set_model(
        create_tensorflow_model(number_of_features, number_of_labels)
    )
    #!!! my_model_handler.desired_outputs()
    #!!! my_model_handler.training_parameters()
    #!!! my_model_handler.all_labels()
    #!!! my_model_handler.some_labels()
    #!!! my_model_handler.train()
    #!!! my_model_handler.predict()


"""
Create a toy set of features
"""


def create_dataset(number_of_superpixels, number_of_features):
    import numpy as np

    my_features = np.random.normal(
        0, 1, size=(number_of_superpixels, number_of_features)
    )

    # Create labels: Unknown=0. Known=1, ..., number_of_labels.  (Note that we could
    # instead use different numbers, or strings, etc.)
    labels_per_superpixel = 1  # May need more if a GAN, etc.
    number_of_labels = 3
    label_names = ["Unknown"] + [
        "Label" + str(i) for i in range(1, number_of_labels + 1)
    ]
    my_label_definitions = {
        idx: {"description": name} for idx, name in enumerate(label_names)
    }
    # Create a random label for each superpixel
    my_labels = np.random.randint(
        1,
        number_of_labels + 1,
        size=(number_of_superpixels, labels_per_superpixel),
    )
    return number_of_labels, my_features, my_label_definitions, my_labels


"""
Create a toy TensorFlow model that has the right shape for inputs and outputs
"""


def create_tensorflow_model(number_of_features, number_of_labels):
    import tensorflow as tf

    model = tf.keras.models.Sequential(
        [
            tf.keras.Input(shape=(number_of_features,)),
            tf.keras.layers.Dense(number_of_labels, activation="relu"),
        ],
        name=f"{number_of_labels}_labels_from_{number_of_features}_features",
    )
    return model


if __name__ == "__main__":
    test_imports()
    test_high_level()
