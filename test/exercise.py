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
from create import create_dataset
from create import create_toy_pytorch_model
from create import create_toy_tensorflow_model
from typing import List


def exercise_dataset_handler(
    my_dataset_handler,
    number_of_superpixels: int,
    number_of_features: int,
    number_of_categories_by_label: List[int],
    label_to_test: int,
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
    assert np.equal(
        my_dataset_handler.get_all_feature_vectors(), np.array((), dtype=np.int64)
    ).all()
    assert np.equal(
        my_dataset_handler.get_all_labels(), np.array((), dtype=np.int64)
    ).all()

    # Create random feature vectors
    my_feature_vectors, my_label_definitions, my_labels = create_dataset(
        number_of_superpixels, number_of_features, number_of_categories_by_label
    )

    my_dataset_handler.set_all_feature_vectors(my_feature_vectors)
    assert my_dataset_handler.get_all_feature_vectors() is my_feature_vectors
    my_dataset_handler.clear_all_feature_vectors()
    assert np.equal(
        my_dataset_handler.get_all_feature_vectors(), np.array((), dtype=np.int64)
    ).all()

    my_dataset_handler.set_all_labels(my_labels)
    assert my_dataset_handler.get_all_labels() is my_labels
    my_dataset_handler.set_all_labels(my_labels[label_to_test])
    assert (my_dataset_handler.get_all_labels() == my_labels[label_to_test]).all()
    my_dataset_handler.clear_all_labels()
    assert np.equal(
        my_dataset_handler.get_all_labels(), np.array((), dtype=np.int64)
    ).all()

    my_dataset_handler.set_all_feature_vectors(my_feature_vectors)
    my_dataset_handler.set_all_labels(my_labels)
    my_dataset_handler.set_all_label_definitions(my_label_definitions)
    my_dataset_handler.check_data_consistency()


def exercise_model_handler(
    my_model_handler,
    number_of_features: int,
    number_of_categories_by_label: List[int],
    label_to_test: int,
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
