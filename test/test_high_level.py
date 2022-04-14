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


def test_high_level():
    """Purpose: Test that high-level operations work"""
    import al_bench as alb
    import numpy as np

    ## Create handlers
    my_dataset_handler = alb.dataset.DatasetHandler()
    my_model_handler = alb.model.TensorFlowModelHandler()
    my_strategy_handler = alb.strategy.StrategyHandler()
    assert my_dataset_handler.get_all_features() is None
    assert my_dataset_handler.get_all_labels() is None
    assert my_strategy_handler.get_model_handler() is None
    assert my_strategy_handler.get_dataset_handler() is None

    # Create feature vectors
    number_of_superpixels = 10
    number_of_features = 6
    my_features = np.random.normal(
        0, 1, size=(number_of_superpixels, number_of_features)
    )

    # Create labels
    min_label, max_label = 0, 3  # inclusive, exclusive
    labels_per_superpixel = 1
    my_labels = np.random.random_integers(
        0, max_label - 1, size=(number_of_superpixels, labels_per_superpixel)
    )
    my_label_definitions = ["Unknown"] + ["Label" + str(i) for i in range(1, max_label)]

    ## Exercise the StrategyHandler
    my_strategy_handler.set_model_handler(my_model_handler)
    assert my_strategy_handler.get_model_handler() is my_model_handler
    my_strategy_handler.clear_model_handler()
    assert my_strategy_handler.get_model_handler() is None
    my_strategy_handler.set_model_handler(my_model_handler)
    assert my_strategy_handler.get_model_handler() is my_model_handler

    my_strategy_handler.set_dataset_handler(my_dataset_handler)
    assert my_strategy_handler.get_dataset_handler() is my_dataset_handler
    my_strategy_handler.clear_dataset_handler()
    assert my_strategy_handler.get_dataset_handler() is None
    my_strategy_handler.set_dataset_handler(my_dataset_handler)
    assert my_strategy_handler.get_dataset_handler() is my_dataset_handler

    #!!! my_strategy_handler.set_scoring_metric()
    #!!! my_strategy_handler.set_diversity_metric()

    ## Exercise the DatasetHandler
    my_dataset_handler.set_all_features(my_features)
    assert my_dataset_handler.get_all_features() is my_features
    my_dataset_handler.clear_all_features()
    assert my_dataset_handler.get_all_features() is None
    my_dataset_handler.set_all_features(my_features)
    assert my_dataset_handler.get_all_features() is my_features
    assert my_strategy_handler.get_dataset_handler().get_all_features() is my_features

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

    ## Exercise the ModelHandler
