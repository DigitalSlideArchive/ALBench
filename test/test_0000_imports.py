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
    from al_bench import dataset  # noqa F401
    from al_bench import model  # noqa F401
    from al_bench import strategy  # noqa F401
    from al_bench.dataset import GenericDatasetHandler  # noqa F401
    from al_bench.model import ModelStep  # noqa F401
    from al_bench.model import PyTorchModelHandler  # noqa F401
    from al_bench.model import TensorFlowModelHandler  # noqa F401
    from al_bench.strategy import EntropyStrategyHandler  # noqa F401
    from al_bench.strategy import LeastConfidenceStrategyHandler  # noqa F401
    from al_bench.strategy import LeastMarginStrategyHandler  # noqa F401
    from al_bench.strategy import RandomStrategyHandler  # noqa F401
    from datetime import datetime  # noqa F401
    from h5py import File  # noqa F401
    from numpy import append  # noqa F401
    from numpy import array  # noqa F401
    from numpy import clip  # noqa F401
    from numpy import float32  # noqa F401
    from numpy import float64  # noqa F401
    from numpy import floor  # noqa F401
    from numpy import ndarray  # noqa F401
    from numpy.random import default_rng  # noqa F401
    from os.path import join  # noqa F401
    from random import sample  # noqa F401
    from re import search  # noqa F401
    from tensorflow.keras import Input  # noqa F401
    from tensorflow.keras.layers import Dense  # noqa F401
    from tensorflow.keras.layers import Dropout  # noqa F401
    from tensorflow.keras.models import Sequential  # noqa F401
    from torch.nn import Dropout  # noqa F401
    from torch.nn import Linear  # noqa F401
    from torch.nn import ReLU  # noqa F401
    from torch.nn import Softmax  # noqa F401
    from torch.nn.modules.module import Module  # noqa F401

    pass


if __name__ == "__main__":
    test_0000_imports()
