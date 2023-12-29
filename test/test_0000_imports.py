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


def test_0000_imports() -> None:
    """Purpose: Test that needed parts of needed packages are available"""
    from al_bench import dataset  # noqa F401
    from al_bench import model  # noqa F401
    from al_bench import strategy  # noqa F401
    from al_bench.dataset import AbstractDatasetHandler  # noqa F401
    from al_bench.dataset import GenericDatasetHandler  # noqa F401
    from al_bench.model import AbstractModelHandler  # noqa F401
    from al_bench.model import ModelStep  # noqa F401
    from al_bench.model import PyTorchModelHandler  # noqa F401
    from al_bench.model import SamplingBayesianPyTorchModelHandler  # noqa F401
    from al_bench.model import SamplingBayesianTensorFlowModelHandler  # noqa F401
    from al_bench.model import TensorFlowModelHandler  # noqa F401
    from al_bench.model import VariationalBayesianPyTorchModelHandler  # noqa F401
    from al_bench.model import VariationalBayesianTensorFlowModelHandler  # noqa F401
    from al_bench.strategy import MaximumEntropyStrategyHandler  # noqa F401
    from al_bench.strategy import LeastConfidenceStrategyHandler  # noqa F401
    from al_bench.strategy import LeastMarginStrategyHandler  # noqa F401
    from al_bench.strategy import RandomStrategyHandler  # noqa F401
    from datetime import datetime  # noqa F401
    from enum import Enum  # noqa F401
    from h5py import File  # noqa F401
    from numpy import amax  # noqa F401
    from numpy import any  # noqa F401
    from numpy import arange  # noqa F401
    from numpy import argsort  # noqa F401
    from numpy import array  # noqa F401
    from numpy import clip  # noqa F401
    from numpy import concatenate  # noqa F401
    from numpy import equal  # noqa F401
    from numpy import float32  # noqa F401
    from numpy import float64  # noqa F401
    from numpy import floor  # noqa F401
    from numpy import full  # noqa F401
    from numpy import isnan  # noqa F401
    from numpy import nan  # noqa F401
    from numpy import ndarray  # noqa F401
    from numpy import newaxis  # noqa F401
    from numpy import percentile  # noqa F401
    from numpy import random  # noqa F401
    from numpy import typing  # noqa F401
    from numpy import zeros  # noqa F401
    from numpy.random import default_rng  # noqa F401
    from numpy.typing import NDArray  # noqa F401
    from os import path  # noqa F401
    from os.path import join  # noqa F401
    from random import sample  # noqa F401
    from re import search  # noqa F401
    from scipy import stats  # noqa F401
    from scipy.stats import entropy  # noqa F401
    from tensorflow import keras  # noqa F401
    from tensorflow.keras import Input  # noqa F401
    from tensorflow.keras import Model  # noqa F401
    from tensorflow.keras import layers  # noqa F401
    from tensorflow.keras import models  # noqa F401
    from tensorflow.keras.callbacks import Callback  # noqa F401
    from tensorflow.keras.layers import Dense  # noqa F401
    from tensorflow.keras.layers import Dropout  # noqa F401
    from tensorflow.keras.losses import SparseCategoricalCrossentropy  # noqa F401
    from tensorflow.keras.models import Sequential  # noqa F401
    from torch import argmax  # noqa F401
    from torch import clamp  # noqa F401
    from torch import eye  # noqa F401
    from torch import from_numpy  # noqa F401
    from torch import load  # noqa F401
    from torch import log  # noqa F401
    from torch import nn  # noqa F401
    from torch import save  # noqa F401
    from torch.nn import Dropout  # noqa F401
    from torch.nn import Linear  # noqa F401
    from torch.nn import ReLU  # noqa F401
    from torch.nn import Softmax  # noqa F401
    from torch.nn.modules import module  # noqa F401
    from torch.nn.modules.module import Module  # noqa F401
    from torch.optim import SGD  # noqa F401
    from torch.utils import tensorboard  # noqa F401
    from torch.utils.data import DataLoader  # noqa F401
    from torch.utils.data import Dataset  # noqa F401
    from torch.utils.tensorboard import SummaryWriter  # noqa F401
    from typing import Dict  # noqa F401
    from typing import Iterable  # noqa F401
    from typing import List  # noqa F401
    from typing import Mapping  # noqa F401
    from typing import MutableMapping  # noqa F401
    from typing import Sequence  # noqa F401

    pass


if __name__ == "__main__":
    test_0000_imports()
