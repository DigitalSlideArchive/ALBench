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

"""Active Learning Benchmarking tool"""

__version__ = "0.0.1"

"""

The tool takes an input dataset, machine learning model, and active learning strategy
and outputs information to be used in evaluating how well the strategy does with that
model and dataset.  By running the tool multiple times with different inputs, the tool
allows comparisons across different active learning strategies and also allows
comparisons across different models and across different datasets.  Researchers can use
the tool to test proposed active learning strategies in the context of a specific model
and dataset; or multiple models and datasets can be used to get a broader picture of
each strategy's effectiveness in multiple contexts.  As an alternative use case,
multiple runs of the tool with different models and datasets can be compared, evaluating
these models and datasets for their compatibility with a given active learning strategy.

"""
from . import dataset, model, strategy
