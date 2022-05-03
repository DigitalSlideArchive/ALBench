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

import numpy as np

"""
BenchMarkTool is a class that can be configured to run an active learning strategy.
At some point we should have a abstract superclass that defines the interface!!!
"""


class BenchmarkTool:
    def __init__(self):
        self.model_handler = None
        self.dataset_handler = None
        self.dataset_handler = None

    def set_model_handler(self, model_handler):
        if model_handler is None:
            raise ValueError("The argument to set_model_handler must not be None")
        self.model_handler = model_handler

    def get_model_handler(self):
        return self.model_handler

    def clear_model_handler(self):
        self.model_handler = None

    def set_dataset_handler(self, dataset_handler):
        if dataset_handler is None:
            raise ValueError("The argument to set_dataset_handler must not be None")
        self.dataset_handler = dataset_handler

    def get_dataset_handler(self):
        return self.dataset_handler

    def clear_dataset_handler(self):
        self.dataset_handler = None

    def set_strategy_handler(self, strategy_handler):
        if strategy_handler is None:
            raise ValueError("The argument to set_strategy_handler must not be None")
        self.strategy_handler = strategy_handler

    def get_strategy_handler(self):
        return self.strategy_handler

    def clear_strategy_handler(self):
        self.strategy_handler = None
