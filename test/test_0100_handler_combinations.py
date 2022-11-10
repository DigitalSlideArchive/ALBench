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
import datetime
import numpy as np
import os
import random
import re
from check import check_deeply_numeric
from create import create_dataset_4598_1280_4
from create import create_pytorch_model_with_dropout
from create import create_tensorflow_model_with_dropout
from numpy.typing import NDArray
from typing import Any, Dict, List, Type


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
    parameters: Dict[str, Any] = {
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
                alb.strategy.EntropyStrategyHandler,
            ):
                # Create fresh handlers and components
                my_feature_vectors: NDArray
                my_label_definitions: List[Dict]
                my_labels: NDArray
                my_feature_vectors, my_label_definitions, my_labels = dataset_creator(
                    **parameters
                )
                my_dataset_handler: alb.dataset.AbstractDatasetHandler = (
                    DatasetHandler()
                )
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

                my_strategy_handler: alb.strategy.AbstractStrategyHandler = (
                    StrategyHandler()
                )
                my_strategy_handler.set_dataset_handler(my_dataset_handler)
                my_strategy_handler.set_model_handler(my_model_handler)
                my_strategy_handler.set_learning_parameters(
                    maximum_queries=number_queries,
                    label_of_interest=parameters["label_to_test"],
                    number_to_select_per_query=number_per_query,
                )

                # Start with nothing labeled yet
                currently_labeled_examples: NDArray = np.array(())

                # Go!
                combination_name: str = "-".join(
                    [
                        # re.search(
                        #     r"<class 'al_bench\.dataset\.(.*)'>",
                        #     f"{type(my_dataset_handler)}",
                        # ).group(1),
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
                ################
                my_strategy_handler.run(currently_labeled_examples)
                ################
                my_log: List = my_strategy_handler.get_log()
                assert isinstance(my_log, list)
                assert all([isinstance(e, dict) for e in my_log])
                assert all([isinstance(e["utcnow"], datetime.datetime) for e in my_log])
                assert all(
                    [isinstance(e["model_step"], alb.model.ModelStep) for e in my_log]
                )
                assert all([isinstance(e["logs"], dict) for e in my_log])
                assert all(
                    [
                        check_deeply_numeric(v)
                        for e in my_log
                        for v in e["logs"].values()
                    ]
                )
                my_strategy_handler.write_train_log_for_tensorboard(
                    log_dir=os.path.join("runs", combination_name)
                )
                my_strategy_handler.write_confidence_log_for_tensorboard(
                    log_dir=os.path.join("runs", combination_name)
                )


if __name__ == "__main__":
    test_0100_handler_combinations()
