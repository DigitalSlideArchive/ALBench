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

import datetime
import os
import random
import re
from typing import Any, List, Mapping, Match, Optional, Type

import numpy as np

import al_bench as alb
import al_bench.strategy
from check import NDArrayFloat, NDArrayInt, check_deeply_numeric
from create import (
    create_dataset_4598_1280_4,
    create_pytorch_model_with_dropout,
    create_tensorflow_model_with_dropout,
)


def test_0100_handler_combinations() -> None:
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

    # Specify some testing parameters
    parameters: Mapping[str, Any]
    parameters = {
        "number_of_superpixels": 4598,
        "number_of_features": 1280,
        "number_of_categories_by_label": [4],
        "label_to_test": 0,
    }
    number_queries: int = 10
    number_per_query: int = 10

    combination_index: int = 0
    DatasetHandler: Type[alb.dataset.AbstractDatasetHandler]
    for dataset_creator, DatasetHandler in (
        # (
        #     create_dataset,
        #     alb.dataset.GenericDatasetHandler,
        # ),
        (create_dataset_4598_1280_4, alb.dataset.GenericDatasetHandler),
    ):
        ModelHandler: Type[alb.model.AbstractModelHandler]
        for model_creator, ModelHandler in (
            # (
            #     create_toy_tensorflow_model,
            #     alb.model.TensorFlowModelHandler,
            # ),
            # (
            #     create_toy_pytorch_model,
            #     alb.model.PyTorchModelHandler,
            # ),
            (create_tensorflow_model_with_dropout, alb.model.TensorFlowModelHandler),
            (create_pytorch_model_with_dropout, alb.model.PyTorchModelHandler),
        ):
            StrategyHandler: Type[alb.strategy.AbstractStrategyHandler]
            for StrategyHandler in (
                alb.strategy.RandomStrategyHandler,
                alb.strategy.LeastConfidenceStrategyHandler,
                alb.strategy.LeastMarginStrategyHandler,
                alb.strategy.MaximumEntropyStrategyHandler,
            ):
                # Create fresh handlers and components
                my_feature_vectors: NDArrayFloat
                my_label_definitions: List[Mapping]
                my_labels: NDArrayInt
                my_feature_vectors, my_label_definitions, my_labels = dataset_creator(
                    **parameters
                )
                my_dataset_handler: alb.dataset.AbstractDatasetHandler
                my_dataset_handler = DatasetHandler()
                my_dataset_handler.set_all_feature_vectors(my_feature_vectors)
                my_dataset_handler.set_all_label_definitions(my_label_definitions)
                my_dataset_handler.set_all_labels(my_labels)
                my_dataset_handler.set_validation_indices(
                    np.array(
                        random.sample(
                            range(my_feature_vectors.shape[0]),
                            my_feature_vectors.shape[0] // 10,
                        )
                    )
                )

                my_model = model_creator(**parameters)
                my_model_handler: alb.model.AbstractModelHandler = ModelHandler()
                my_model_handler.set_model(my_model)

                my_strategy_handler: alb.strategy.AbstractStrategyHandler
                my_strategy_handler = StrategyHandler()
                my_strategy_handler.set_dataset_handler(my_dataset_handler)
                my_strategy_handler.set_model_handler(my_model_handler)
                my_strategy_handler.set_learning_parameters(
                    maximum_queries=number_queries,
                    label_of_interest=parameters["label_to_test"],
                    number_to_select_per_query=number_per_query,
                )

                # Start with nothing labeled yet
                currently_labeled_examples: NDArrayInt = np.array((), dtype=np.int64)

                # dataset_match: Optional[Match[str]]
                # dataset_match = re.search(
                #     r"<class 'al_bench\.dataset\.(.*)'>",
                #     f"{type(my_dataset_handler)}",
                # )
                # dataset_string: str
                # dataset_string = (
                #     "dataset_handler_NOT_FOUND"
                #     if dataset_match is None
                #     else dataset_match.group(1)
                # )
                model_match: Optional[Match[str]]
                model_match = re.search(
                    r"<class 'al_bench\.model\.(.*)'>", f"{type(my_model_handler)}"
                )
                model_string: str
                model_string = (
                    "model_handler_NOT_FOUND"
                    if model_match is None
                    else model_match.group(1)
                )
                strategy_match: Optional[Match[str]]
                strategy_match = re.search(
                    r"<class 'al_bench\.strategy\.(.*)'>",
                    f"{type(my_strategy_handler)}",
                )
                strategy_string: str
                strategy_string = (
                    "strategy_handler_NOT_FOUND"
                    if strategy_match is None
                    else strategy_match.group(1)
                )
                combination_name: str
                combination_name = "-".join(
                    [
                        # dataset_string,
                        model_string,
                        strategy_string,
                        datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S.%f"),
                        f"{combination_index:06d}",
                    ]
                )

                # Go!
                print(f"Exercise combination: {combination_name}")
                combination_index += 1
                ################
                my_strategy_handler.run(currently_labeled_examples)
                ################
                my_log: List = my_strategy_handler.get_log()
                assert isinstance(my_log, list)
                assert all(isinstance(e, dict) for e in my_log)
                assert all(isinstance(e["utcnow"], datetime.datetime) for e in my_log)
                assert all(
                    isinstance(e["model_step"], alb.model.ModelStep) for e in my_log
                )
                assert all(isinstance(e["logs"], dict) for e in my_log)
                assert all(
                    check_deeply_numeric(v) for e in my_log for v in e["logs"].values()
                )
                my_strategy_handler.write_train_log_for_tensorboard(
                    log_dir=os.path.join("runs", combination_name)
                )
                my_strategy_handler.write_certainty_log_for_tensorboard(
                    log_dir=os.path.join("runs", combination_name)
                )


if __name__ == "__main__":
    test_0100_handler_combinations()
