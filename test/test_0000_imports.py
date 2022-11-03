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


def test_0000_imports():
    """Purpose: Test that needed parts of needed packages are available"""
    from al_bench import dataset
    from al_bench import model
    from al_bench import strategy
    from al_bench.dataset import GenericDatasetHandler
    from al_bench.model import ModelStep
    from al_bench.model import PyTorchModelHandler
    from al_bench.model import TensorFlowModelHandler
    from al_bench.strategy import EntropyStrategyHandler
    from al_bench.strategy import LeastConfidenceStrategyHandler
    from al_bench.strategy import LeastMarginStrategyHandler
    from al_bench.strategy import RandomStrategyHandler
    from datetime import datetime
    from h5py import File
    from numpy import append
    from numpy import array
    from numpy import clip
    from numpy import float32
    from numpy import float64
    from numpy import floor
    from numpy import ndarray
    from numpy.random import default_rng
    from os.path import join
    from random import sample
    from re import search
    from tensorflow.keras import Input
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.models import Sequential
    from torch.nn import Dropout
    from torch.nn import Linear
    from torch.nn import ReLU
    from torch.nn import Softmax
    from torch.nn.modules.module import Module

    pass


if __name__ == "__main__":
    test_0000_imports()
