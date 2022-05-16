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
from . import dataset

"""
AbstractModelHandler is an abstract class that defines the interface for a machine
learning model.
"""


class AbstractModelHandler:
    def set_model(self, model):
        """
        Set the underlying model handled by this model handler.  This is generally
        called just once.
        """
        raise NotImplementedError(
            "Abstract method AbstractModelHandler::set_model should not be called."
        )

    def set_training_parameters(self):
        """
        Set the training parameters needed by the underlying model.  This is generally
        called just once and generally includes batch size, number of epochs, loss
        function, and stopping conditions.
        """
        raise NotImplementedError(
            "Abstract method AbstractModelHandler::set_training_parameters "
            "should not be called."
        )

    def set_dataset_handler(self, dataset_handler):
        raise NotImplementedError(
            "Abstract method AbstractModelHandler::set_dataset_handler "
            "should not be called."
        )

    def get_dataset_handler(self):
        raise NotImplementedError(
            "Abstract method AbstractModelHandler::get_dataset_handler "
            "should not be called."
        )

    def train(self, train_features, train_labels):
        """
        Ask the model to train.  This is generally called each time new labels have been
        provided.  Add training weights!!!
        """
        raise NotImplementedError(
            "Abstract method AbstractModelHandler::train should not be called."
        )

    def predict(self, features):
        """
        Ask the model to make predictions.  This is generally called after training so
        that the strategy can show the user the new predictions and ask for corrections.
        Parameters include which examples should be predicted.
        """
        raise NotImplementedError(
            "Abstract method AbstractModelHandler::predict should not be called."
        )


class GenericModelHandler(AbstractModelHandler):
    """
    GenericModelHandler handles functionality that is agnostic to the choice of the
    deep-learning framework.  GenericModelHandler is the superclass for
    TensorFlowModelHandler, PyTorchModelHandler, etc.  These subclasses handle
    GenericModelHandler operations that are dependent upon which deep learning framework
    is being used.
    """

    def __init__(self):
        self.dataset_handler = None

    def set_dataset_handler(self, dataset_handler):
        if not isinstance(dataset_handler, dataset.AbstractDatasetHandler):
            raise ValueError(
                "The argument to set_dataset_handler must be a (subclass of) "
                "AbstractDatasetHandler"
            )
        self.dataset_handler = dataset_handler

    def get_dataset_handler(self):
        return self.dataset_handler


class TensorFlowModelHandler(GenericModelHandler):
    """
    TensorFlowModelHandler is the class that implements framework-agnostic
    GenericModelHandler routines via TensorFlow.
    """

    def __init__(self):
        self.model = None

    def set_model(self, model):
        """
        Set the underlying model handled by this model handler.  This is generally
        called just once.
        """
        import tensorflow as tf

        if not isinstance(model, tf.keras.Model):
            raise ValueError(
                "The parameter of TensorFlowModelHandler.set_model must be "
                "tf.keras.Model"
            )
        self.model = model

    def set_training_parameters(self):
        """
        Set the training parameters needed by the underlying model.  This is generally
        called just once and generally includes batch size, number of epochs, loss
        function, and stopping conditions.
        """
        raise NotImplementedError("Not implemented")

    def train(self, train_features, train_labels):
        """
        Ask the model to train.  This is generally called each time new labels have been
        provided.  Add training weights!!!
        """
        import tensorflow as tf

        assert not np.any(np.isnan(train_features))
        assert not np.any(np.isnan(train_labels))

        self.model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )
        #!!! Get `epochs` from training parameters.
        self.model.fit(train_features, train_labels, epochs=10)

    def predict(self, features):
        """
        Ask the model to make predictions.  This is generally called after training so
        that the strategy can show the user the new predictions and ask for corrections.
        Parameters include which examples should be predicted.
        """

        predictions = self.model.predict(features)
        return predictions


class PyTorchModelHandler(GenericModelHandler):
    """
    PyTorchModelHandler is the class that implements framework-agnostic
    GenericModelHandler routines via PyTorch.
    """

    def __init__(self):
        raise NotImplementedError("Not implemented")

    def set_model(self):
        """
        Set the underlying model handled by this model handler.  This is generally
        called just once.
        """
        raise NotImplementedError("Not implemented")

    def set_training_parameters(self):
        """
        Set the training parameters needed by the underlying model.  This is generally
        called just once and generally includes batch size, number of epochs, loss
        function, and stopping conditions.
        """
        raise NotImplementedError("Not implemented")

    def train(self, train_features, train_labels):
        """
        Ask the model to train.  This is generally called each time new labels have been
        provided.  Add training weights!!!
        """
        raise NotImplementedError("Not implemented")

    def predict(self, features):
        """
        Ask the model to make predictions.  This is generally called after training so
        that the strategy can show the user the new predictions and ask for corrections.
        Parameters include which examples should be predicted.
        """
        raise NotImplementedError("Not implemented")


class AbstractEnsembleModelHandler(GenericModelHandler):
    """
    AbstractEnsembleModelHandler is a generic implementation of a GenericModelHandler
    that uses a committee of modelHandlers, via voting or similar, to determine its
    responses.  It's subclasses deliver an actual implementation.
    """

    def __init__(self):
        raise NotImplementedError("Not implemented")

    def set_model(self):
        """
        Set the underlying model handled by this model handler.  This is generally
        called just once.
        """
        raise NotImplementedError("Not implemented")

    def set_training_parameters(self):
        """
        Set the training parameters needed by the underlying model.  This is generally
        called just once and generally includes batch size, number of epochs, loss
        function, and stopping conditions.
        """
        raise NotImplementedError("Not implemented")

    def train(self, train_features, train_labels):
        """
        Ask the model to train.  This is generally called each time new labels have been
        provided.  Add training weights!!!
        """
        raise NotImplementedError("Not implemented")

    def predict(self, features):
        """
        Ask the model to make predictions.  This is generally called after training so
        that the strategy can show the user the new predictions and ask for corrections.
        Parameters include which examples should be predicted.
        """
        raise NotImplementedError("Not implemented")


class ExampleEnsembleModelHandler(AbstractEnsembleModelHandler):
    """
    ExampleEnsembleModelHandler is a specific implementation of a
    AbstractEnsembleModelHandler that uses a committee of modelHandlers, via voting or
    similar, to determine its responses.  The methods of ExampleEnsembleModelHandler
    determine the specific implementation of this ensemble.
    """

    def __init__(self):
        raise NotImplementedError("Not implemented")

    def set_model(self):
        """
        Set the underlying model handled by this model handler.  This is generally
        called just once.
        """
        raise NotImplementedError("Not implemented")

    def set_training_parameters(self):
        """
        Set the training parameters needed by the underlying model.  This is generally
        called just once and generally includes batch size, number of epochs, loss
        function, and stopping conditions.
        """
        raise NotImplementedError("Not implemented")

    def train(self, train_features, train_labels):
        """
        Ask the model to train.  This is generally called each time new labels have been
        provided.  Add training weights!!!
        """
        raise NotImplementedError("Not implemented")

    def predict(self, features):
        """
        Ask the model to make predictions.  This is generally called after training so
        that the strategy can show the user the new predictions and ask for corrections.
        Parameters include which examples should be predicted.
        """
        raise NotImplementedError("Not implemented")
