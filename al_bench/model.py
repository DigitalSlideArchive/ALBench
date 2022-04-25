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
import tensorflow as tf


class GenericModelHandler:
    """
    Functionality that is agnostic to the choice of the deep-learning framework is
    handled by this class.  GenericModelHandler is the superclass for
    TensorFlowModelHandler, PyTorchModelHandler, etc.  These subclasses handle
    GenericModelHandler operations that are dependent upon which deep learning framework
    is being used.
    """

    def __init__():
        pass

    def set_model(self, model):
        raise ValueError(
            "Abstract method GenericModel::set_model should not be called."
        )

    def set_desired_outputs(self):
        # Write me!!!  Alternatively, record everything and then have a method that
        # selectively fetches some of them.  This may be called from subclass overrides.
        pass

    def set_training_parameters(self):
        # Write me!!!  Generally called just once.  Includes batch size, number of
        # epochs, loss function / stopping conditions.
        pass

    def set_all_labels(self):
        # Write me!!!  Be able to handle different treatments for newly labeled
        # examples; generally, handle a weighting scheme that depends upon when examples
        # were labeled.
        pass

    def set_some_labels(self):
        # Write me!!!  Be able to handle different treatments for newly labeled
        # examples; generally, handle a weighting scheme that depends upon when examples
        # were labeled.
        pass

    def train(self):
        # Write me!!!  Generally called each time we train.  Includes dataset; should
        # this already be divided into train, test, verify?  Should it already have
        # weights?
        pass

    def predict(self):
        # Write me!!! Which members of the data set should we predict for?
        pass


class TensorFlowModelHandler(GenericModelHandler):
    """
    TensorFlowModelHandler is the class that implements framework-agnostic
    GenericModelHandler routines via TensorFlow.
    """

    def __init__(self):
        self.model = None

    def set_model(self, model):
        if not isinstance(model, tf.keras.Model):
            raise ValueError(
                "The parameter of TensorFlowModelHandler.set_model must be tf.keras.Model"
            )
        self.model = model


class PyTorchModelHandler(GenericModelHandler):
    """
    PyTorchModelHandler is the class that implements framework-agnostic
    GenericModelHandler routines via PyTorch.
    """

    def __init__(self):
        pass


class GenericEnsembleModelHandler(GenericModelHandler):
    """
    GenericEnsembleModelHandler is a generic implementation of a GenericModelHandler
    that uses a committee of modelHandlers, via voting or similar, to determine its
    responses.  It's subclasses deliver an actual implementation.
    """

    def __init__(self):
        pass


class ExampleEnsembleModelHandler(GenericEnsembleModelHandler):
    """
    ExampleEnsembleModelHandler is a specific implementation of a
    GenericEnsembleModelHandler that uses a committee of modelHandlers, via voting or
    similar, to determine its responses.  The methods of ExampleEnsembleModelHandler
    determine the specific implementation of this ensemble.
    """

    def __init__(self):
        pass
