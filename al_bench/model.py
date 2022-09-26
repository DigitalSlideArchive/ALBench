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

from datetime import datetime
import enum
import numpy as np
from . import dataset


class ModelStep(enum.Enum):
    ON_TRAIN_BEGIN = 100
    ON_TRAIN_END = 105
    ON_TRAIN_EPOCH_BEGIN = 120
    ON_TRAIN_EPOCH_END = 125
    ON_TRAIN_BATCH_BEGIN = 140
    ON_TRAIN_BATCH_END = 145
    ON_TEST_BEGIN = 200
    ON_TEST_END = 205
    ON_TEST_BATCH_BEGIN = 240
    ON_TEST_BATCH_END = 245
    ON_PREDICT_BEGIN = 300
    ON_PREDICT_END = 305
    ON_PREDICT_BATCH_BEGIN = 340
    ON_PREDICT_BATCH_END = 345


class AbstractModelHandler:
    """
    AbstractModelHandler is an abstract class that defines the interface for a machine
    learning model.
    """

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
    AbstractModelHandler operations that are dependent upon which deep learning
    framework is being used.
    """

    def __init__(self):
        super(GenericModelHandler, self).__init__()
        self.dataset_handler = None
        self.custom_callback = GenericModelHandler.CustomCallback()

    def reset_log(self):
        self.custom_callback.reset_log()

    def get_log(self):
        return self.custom_callback.get_log()

    def set_dataset_handler(self, dataset_handler):
        if not isinstance(dataset_handler, dataset.AbstractDatasetHandler):
            raise ValueError(
                "The argument to set_dataset_handler must be a (subclass of) "
                "AbstractDatasetHandler"
            )
        self.dataset_handler = dataset_handler

    def get_dataset_handler(self):
        return self.dataset_handler

    import keras

    class CustomCallback(keras.callbacks.Callback):
        def __init__(self):
            super(GenericModelHandler.CustomCallback, self).__init__()
            self.reset_log()

        def reset_log(self):
            self.log = list()

        def get_log(self):
            return self.log

        def on_train_begin(self, logs=None):
            self.log.append(
                dict(
                    utcnow=datetime.utcnow(),
                    model_step=ModelStep.ON_TRAIN_BEGIN,
                    logs=logs,
                )
            )

        def on_train_end(self, logs=None):
            self.log.append(
                dict(
                    utcnow=datetime.utcnow(),
                    model_step=ModelStep.ON_TRAIN_END,
                    logs=logs,
                )
            )

        def on_epoch_begin(self, epoch, logs=None):
            self.log.append(
                dict(
                    utcnow=datetime.utcnow(),
                    model_step=ModelStep.ON_TRAIN_EPOCH_BEGIN,
                    epoch=epoch,
                    logs=logs,
                )
            )

        def on_epoch_end(self, epoch, logs=None):
            self.log.append(
                dict(
                    utcnow=datetime.utcnow(),
                    model_step=ModelStep.ON_TRAIN_EPOCH_END,
                    epoch=epoch,
                    logs=logs,
                )
            )

        def on_train_batch_begin(self, batch, logs=None):
            self.log.append(
                dict(
                    utcnow=datetime.utcnow(),
                    model_step=ModelStep.ON_TRAIN_BATCH_BEGIN,
                    batch=batch,
                    logs=logs,
                )
            )

        def on_train_batch_end(self, batch, logs=None):
            # For tensorflow, logs.keys() == ["loss", "accuracy"]
            self.log.append(
                dict(
                    utcnow=datetime.utcnow(),
                    model_step=ModelStep.ON_TRAIN_BATCH_END,
                    batch=batch,
                    logs=logs,
                )
            )

        def on_test_begin(self, logs=None):
            self.log.append(
                dict(
                    utcnow=datetime.utcnow(),
                    model_step=ModelStep.ON_TEST_BEGIN,
                    logs=logs,
                )
            )

        def on_test_end(self, logs=None):
            self.log.append(
                dict(
                    utcnow=datetime.utcnow(),
                    model_step=ModelStep.ON_TEST_END,
                    logs=logs,
                )
            )

        def on_test_batch_begin(self, batch, logs=None):
            self.log.append(
                dict(
                    utcnow=datetime.utcnow(),
                    model_step=ModelStep.ON_TEST_BATCH_BEGIN,
                    batch=batch,
                    logs=logs,
                )
            )

        def on_test_batch_end(self, batch, logs=None):
            self.log.append(
                dict(
                    utcnow=datetime.utcnow(),
                    model_step=ModelStep.ON_TEST_BATCH_END,
                    batch=batch,
                    logs=logs,
                )
            )

        def on_predict_begin(self, logs=None):
            self.log.append(
                dict(
                    utcnow=datetime.utcnow(),
                    model_step=ModelStep.ON_PREDICT_BEGIN,
                    logs=logs,
                )
            )

        def on_predict_end(self, logs=None):
            self.log.append(
                dict(
                    utcnow=datetime.utcnow(),
                    model_step=ModelStep.ON_PREDICT_END,
                    logs=logs,
                )
            )

        def on_predict_batch_begin(self, batch, logs=None):
            self.log.append(
                dict(
                    utcnow=datetime.utcnow(),
                    model_step=ModelStep.ON_PREDICT_BATCH_BEGIN,
                    batch=batch,
                    logs=logs,
                )
            )

        def on_predict_batch_end(self, batch, logs=None):
            # For tensorflow, logs.keys() == ["outputs"]
            self.log.append(
                dict(
                    utcnow=datetime.utcnow(),
                    model_step=ModelStep.ON_PREDICT_BATCH_END,
                    batch=batch,
                    logs=logs,
                )
            )


class TensorFlowModelHandler(GenericModelHandler):
    """
    TensorFlowModelHandler is the class that implements framework-dependent
    GenericModelHandler routines via TensorFlow.
    """

    def __init__(self):
        import tensorflow as tf

        super(TensorFlowModelHandler, self).__init__()
        self.model = None
        self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False
        )

    def set_model(self, model):
        """
        Set the underlying model handled by this model handler.  This is generally
        called just once.
        """
        import tensorflow as tf

        if not isinstance(model, tf.keras.Model):
            raise ValueError(
                "The parameter of TensorFlowModelHandler.set_model must be of type "
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
        assert not np.any(np.isnan(train_features))
        assert not np.any(np.isnan(train_labels))

        self.model.compile(
            optimizer="adam", loss=self.loss_function, metrics=["accuracy"]
        )
        # Get `epochs` from training parameters!!!
        # Add validation_data=(x, y,) to the self.model.fit call!!!

        self.model.fit(
            train_features,
            train_labels,
            epochs=10,
            verbose=0,
            callbacks=[self.custom_callback],
        )
        # print(f"{repr(self.custom_callback.get_log()) = }")

    def predict(self, features):
        """
        Ask the model to make predictions.  This is generally called after training so
        that the strategy can show the user the new predictions and ask for corrections.
        Parameters include which examples should be predicted.
        """

        predictions = self.model.predict(
            features, verbose=0, callbacks=[self.custom_callback]
        )
        # print(f"{repr(self.custom_callback.get_log()) = }")
        return predictions


class PyTorchModelHandler(GenericModelHandler):
    """
    PyTorchModelHandler is the class that implements framework-dependent
    GenericModelHandler routines via PyTorch.
    """

    def __init__(self):
        import torch

        super(PyTorchModelHandler, self).__init__()
        self.model = None

        def categorical_cross_entropy(y_pred, y_true):
            y_pred = torch.clamp(y_pred, 1e-9, 1 - 1e-9)
            y_true = torch.eye(y_pred.shape[-1])[y_true]
            return -(y_true * torch.log(y_pred)).sum(dim=1).mean()

        self.criterion = categorical_cross_entropy

    def set_model(self, model):
        """
        Set the underlying model handled by this model handler.  This is generally
        called just once.
        """
        import torch

        if not isinstance(model, torch.nn.modules.module.Module):
            raise ValueError(
                "The parameter of PyTorchModelHandler.set_model must be of type "
                "torch.nn.modules.module.Module"
            )
        self.model = model
        # Useful for another day?
        # torch.save(self.model.state_dict(), PATH)
        # self.model.load_state_dict(torch.load(PATH))

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
        import torch

        assert not np.any(np.isnan(train_features))
        assert not np.any(np.isnan(train_labels))

        # This code heavily mimics
        # https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html.  For a
        # more detailed training loop example, see
        # https://towardsdatascience.com/a-tale-of-two-frameworks-985fa7fcec.

        self.custom_callback.on_train_begin()

        # Get `epochs` from training parameters!!!
        number_of_epochs = 3
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

        class ZipDataset(torch.utils.data.Dataset):
            def __init__(self, train_features, train_labels):
                super(ZipDataset, self).__init__()
                self.train_features = torch.from_numpy(train_features)
                self.train_labels = torch.from_numpy(train_labels)

            def __len__(self):
                return self.train_labels.shape[0]

            def __getitem__(self, index):
                return self.train_features[index, :], self.train_labels[index]

        features_labels = ZipDataset(train_features, train_labels)
        # Instead, get `batch_size` from somewhere!!!
        batch_size = 1
        # DataLoader has additional parameters that we may wish to use!!!
        my_data_loader = torch.utils.data.DataLoader(
            features_labels, batch_size=batch_size
        )

        print_interval = -int(-len(my_data_loader) // 4)
        for epoch in range(number_of_epochs):  # loop over the dataset multiple times
            self.custom_callback.on_epoch_begin(epoch)
            running_loss = 0.0
            running_size = 0
            running_correct = 0.0
            for i, data in enumerate(my_data_loader):
                self.custom_callback.on_train_batch_begin(i)
                inputs, labels = data
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                new_size = inputs.size(0)
                running_size += new_size
                new_loss = loss.item() * inputs.size(0)
                running_loss += new_loss
                new_correct = (torch.argmax(outputs, dim=1) == labels).float().sum()
                running_correct += new_correct
                logs = dict(
                    loss=new_loss / new_size,
                    accuracy=(new_correct / new_size).detach().cpu().numpy(),
                )
                self.custom_callback.on_train_batch_end(i, logs)
            logs = dict(
                loss=running_loss / running_size,
                accuracy=(running_correct / running_size).detach().cpu().numpy(),
            )
            self.custom_callback.on_epoch_end(epoch, logs)
        self.custom_callback.on_train_end(logs)  # `logs` is from the last epoch

    def predict(self, features):
        """
        Ask the model to make predictions.  This is generally called after training so
        that the strategy can show the user the new predictions and ask for corrections.
        Parameters include which examples should be predicted.
        """
        import torch

        self.custom_callback.on_predict_begin()
        predictions = self.model(torch.from_numpy(features))
        predictions = predictions.detach().cpu().numpy()
        logs = dict(outputs=predictions)
        self.custom_callback.on_predict_end(logs)
        return predictions


class AbstractEnsembleModelHandler(GenericModelHandler):
    """
    AbstractEnsembleModelHandler is a generic implementation of a GenericModelHandler
    that uses a committee of modelHandlers, via voting or similar, to determine its
    responses.  It's subclasses deliver an actual implementation.
    """

    def __init__(self):
        # super(AbstractEnsembleModelHandler, self).__init__()
        # raise NotImplementedError("Not implemented")
        pass

    def set_model(self, model):
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
        super(ExampleEnsembleModelHandler, self).__init__()
        # raise NotImplementedError("Not implemented")

    def set_model(self, model):
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
