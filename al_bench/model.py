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
from datetime import datetime
import enum
import numpy as np
import scipy.stats
import tensorflow as tf
from numpy.typing import NDArray
from typing import Dict, List, Mapping, MutableMapping, Sequence


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


# !!! Does it matter that this model.Logger code requires `import tensorflow.keras`
# !!! because of its required to install tensorflow.  superclass?  In particular,
# !!! someone who is using only torch might nonetheless be

# !!! Does it matter that this model.Logger code requires `import torch` because of the
# !!! use of torch.utils.tensorboard.SummaryWriter in
# !!! Logger.write_some_log_for_tensorboard?  In particular, someone who is using only
# !!! tensorflow might nonetheless be required to install torch.


class Logger(tf.keras.callbacks.Callback):
    def __init__(self, model_handler: AbstractModelHandler) -> None:
        super(Logger, self).__init__()
        self.reset_log()
        self.training_size: int = 0
        self.model_handler: AbstractModelHandler = model_handler

    def reset_log(self) -> None:
        self.log: List = list()

    def get_log(self) -> List:
        return self.log

    def write_some_log_for_tensorboard(
        self,
        model_steps: Sequence[ModelStep],
        y_dictionary: Mapping,
        x_key: str,
        *args,
        **kwargs,
    ) -> bool:
        from torch.utils.tensorboard import SummaryWriter

        beginning = datetime.utcfromtimestamp(0)
        with SummaryWriter(*args, **kwargs) as writer:
            for entry in self.log:
                logs: Mapping = entry["logs"]
                if entry["model_step"] not in model_steps or len(logs) == 0:
                    continue
                utc_seconds = (entry["utcnow"] - beginning).total_seconds()
                x_value: float = entry[x_key]
                for key in logs.keys() & y_dictionary.keys():
                    name: str = y_dictionary[key]
                    y_value: float = logs[key]
                    """
                    print(
                        "Invoking writer.add_scalar("
                        f"tag={repr(name)}, "
                        f"scalar_value={repr(y_value)}, "
                        f"global_step={repr(x_value)}, "
                        f"walltime={repr(utc_seconds)}, "
                        f"new_style={repr(True)})"
                    )
                    """
                    writer.add_scalar(
                        tag=name,
                        scalar_value=y_value,
                        global_step=x_value,
                        walltime=utc_seconds,
                        new_style=True,
                    )
        return True

    def write_train_log_for_tensorboard(self, *args, **kwargs) -> bool:
        model_steps = (ModelStep.ON_TRAIN_END,)
        y_dictionary: Mapping = {
            "loss": "Loss/train",
            "val_loss": "Loss/validation",
            "accuracy": "Accuracy/train",
            "val_accuracy": "Accuracy/validation",
        }
        x_key: str = "training_size"
        return self.write_some_log_for_tensorboard(
            model_steps, y_dictionary, x_key, *args, **kwargs
        )

    def write_epoch_log_for_tensorboard(self, *args, **kwargs) -> bool:
        model_steps = (ModelStep.ON_TRAIN_EPOCH_END,)
        y_dictionary: Mapping = {
            "loss": "Loss/train",
            "val_loss": "Loss/test",
            "accuracy": "Accuracy/train",
            "val_accuracy": "Accuracy/test",
        }
        x_key: str = "epoch"
        return self.write_some_log_for_tensorboard(
            model_steps, y_dictionary, x_key, *args, **kwargs
        )

    def write_confidence_log_for_tensorboard(self, *args, **kwargs) -> bool:
        if len(self.log) == 0:
            return False
        x_key: str = "training_size"
        beginning = datetime.utcfromtimestamp(0)
        percentiles: Sequence[float] = (5, 10, 25, 50)
        predictions_list: List[NDArray] = list()

        from torch.utils.tensorboard import SummaryWriter

        with SummaryWriter(*args, **kwargs) as writer:
            for entry in self.log:
                model_step = entry["model_step"]

                if model_step == ModelStep.ON_PREDICT_BEGIN:
                    # Clear accumulated records
                    predictions_list = list()

                if model_step in (
                    ModelStep.ON_PREDICT_BATCH_END,
                    ModelStep.ON_PREDICT_END,
                ):
                    # Accumulate any records
                    if (
                        entry.get("logs") is not None
                        and entry.get("logs").get("outputs") is not None
                        and len(entry["logs"]["outputs"]) > 0
                    ):
                        predictions_list.append(entry["logs"]["outputs"])

                if model_step == ModelStep.ON_PREDICT_END:
                    # Compute and write out statics from accumulated predictions
                    if len(predictions_list) == 0:
                        continue

                    predictions: NDArray = np.concatenate(predictions_list, axis=0)
                    predictions_list = list()
                    statistcs: Mapping = (
                        self.model_handler.compute_confidence_statistics(
                            predictions, percentiles
                        )
                    )
                    # Report percentile scores
                    x_value: float = entry[x_key]
                    utc_seconds = (entry["utcnow"] - beginning).total_seconds()
                    for statistic_kind, statistic_percentiles in statistcs.items():
                        for percentile, y_value in statistic_percentiles.items():
                            name: str = f"Confidence/{statistic_kind}/{percentile:02d}%"
                            """
                            print(
                                "Invoking writer.add_scalar("
                                f"tag={repr(name)}, "
                                f"scalar_value={repr(y_value)}, "
                                f"global_step={repr(x_value)}, "
                                f"walltime={repr(utc_seconds)}, "
                                f"new_style={repr(True)})"
                            )
                            """
                            writer.add_scalar(
                                tag=name,
                                scalar_value=y_value,
                                global_step=x_value,
                                walltime=utc_seconds,
                                new_style=True,
                            )
        return True

    # Because tensorflow defines the interface for on_train_begin, etc. for us and
    # invokes it for us, we cannot simply supply training_size through this
    # interface.  Instead we grab it from self.training_size and require that the
    # user has already set that to something reasonable.
    def on_train_begin(self, logs: Dict = dict()) -> None:
        self.log.append(
            {
                "utcnow": datetime.utcnow(),
                "model_step": ModelStep.ON_TRAIN_BEGIN,
                "training_size": self.training_size,
                "logs": logs,
            }
        )

    def on_train_end(self, logs: Dict = dict()) -> None:
        self.log.append(
            {
                "utcnow": datetime.utcnow(),
                "model_step": ModelStep.ON_TRAIN_END,
                "training_size": self.training_size,
                "logs": logs,
            }
        )

    def on_epoch_begin(self, epoch: int, logs: Dict = dict()) -> None:
        self.log.append(
            {
                "utcnow": datetime.utcnow(),
                "model_step": ModelStep.ON_TRAIN_EPOCH_BEGIN,
                "training_size": self.training_size,
                "epoch": epoch,
                "logs": logs,
            }
        )

    def on_epoch_end(self, epoch: int, logs: Dict = dict()) -> None:
        self.log.append(
            {
                "utcnow": datetime.utcnow(),
                "model_step": ModelStep.ON_TRAIN_EPOCH_END,
                "training_size": self.training_size,
                "epoch": epoch,
                "logs": logs,
            }
        )

    def on_train_batch_begin(self, batch: int, logs: Dict = dict()) -> None:
        self.log.append(
            {
                "utcnow": datetime.utcnow(),
                "model_step": ModelStep.ON_TRAIN_BATCH_BEGIN,
                "training_size": self.training_size,
                "batch": batch,
                "logs": logs,
            }
        )

    def on_train_batch_end(self, batch: int, logs: Dict = dict()) -> None:
        # For tensorflow, logs.keys() == ["loss", "accuracy"]
        self.log.append(
            {
                "utcnow": datetime.utcnow(),
                "model_step": ModelStep.ON_TRAIN_BATCH_END,
                "training_size": self.training_size,
                "batch": batch,
                "logs": logs,
            }
        )

    def on_test_begin(self, logs: Dict = dict()) -> None:
        self.log.append(
            {
                "utcnow": datetime.utcnow(),
                "model_step": ModelStep.ON_TEST_BEGIN,
                "training_size": self.training_size,
                "logs": logs,
            }
        )

    def on_test_end(self, logs: Dict = dict()) -> None:
        self.log.append(
            {
                "utcnow": datetime.utcnow(),
                "model_step": ModelStep.ON_TEST_END,
                "training_size": self.training_size,
                "logs": logs,
            }
        )

    def on_test_batch_begin(self, batch: int, logs: Dict = dict()) -> None:
        self.log.append(
            {
                "utcnow": datetime.utcnow(),
                "model_step": ModelStep.ON_TEST_BATCH_BEGIN,
                "training_size": self.training_size,
                "batch": batch,
                "logs": logs,
            }
        )

    def on_test_batch_end(self, batch: int, logs: Dict = dict()) -> None:
        self.log.append(
            {
                "utcnow": datetime.utcnow(),
                "model_step": ModelStep.ON_TEST_BATCH_END,
                "training_size": self.training_size,
                "batch": batch,
                "logs": logs,
            }
        )

    def on_predict_begin(self, logs: Dict = dict()) -> None:
        self.log.append(
            {
                "utcnow": datetime.utcnow(),
                "model_step": ModelStep.ON_PREDICT_BEGIN,
                "training_size": self.training_size,
                "logs": logs,
            }
        )

    def on_predict_end(self, logs: Dict = dict()) -> None:
        self.log.append(
            {
                "utcnow": datetime.utcnow(),
                "model_step": ModelStep.ON_PREDICT_END,
                "training_size": self.training_size,
                "logs": logs,
            }
        )

    def on_predict_batch_begin(self, batch: int, logs: Dict = dict()) -> None:
        self.log.append(
            {
                "utcnow": datetime.utcnow(),
                "model_step": ModelStep.ON_PREDICT_BATCH_BEGIN,
                "training_size": self.training_size,
                "batch": batch,
                "logs": logs,
            }
        )

    def on_predict_batch_end(self, batch: int, logs: Dict = dict()) -> None:
        # For tensorflow, logs.keys() == ["outputs"]
        self.log.append(
            {
                "utcnow": datetime.utcnow(),
                "model_step": ModelStep.ON_PREDICT_BATCH_END,
                "training_size": self.training_size,
                "batch": batch,
                "logs": logs,
            }
        )


class AbstractModelHandler:
    """
    AbstractModelHandler is an abstract class that defines the interface for a machine
    learning model.
    """

    def __init__(self) -> None:
        raise NotImplementedError(
            "Abstract method AbstractModelHandler::__init__ should not be called."
        )
        self.logger: Logger

    def reset_log(self: AbstractModelHandler) -> None:
        raise NotImplementedError(
            "Abstract method AbstractModelHandler::reset_log should not be called."
        )

    def get_log(self: AbstractModelHandler) -> List:
        raise NotImplementedError(
            "Abstract method AbstractModelHandler::get_log should not be called."
        )

    def get_logger(self: AbstractModelHandler) -> Logger:
        raise NotImplementedError(
            "Abstract method AbstractModelHandler::get_log should not be called."
        )

    def write_train_log_for_tensorboard(
        self: AbstractModelHandler, *args, **kwargs
    ) -> bool:
        raise NotImplementedError(
            "Abstract method AbstractModelHandler::write_train_log_for_tensorboard "
            "should not be called."
        )

    def write_epoch_log_for_tensorboard(
        self: AbstractModelHandler, *args, **kwargs
    ) -> bool:
        raise NotImplementedError(
            "Abstract method AbstractModelHandler::write_epoch_log_for_tensorboard "
            "should not be called."
        )

    def write_confidence_log_for_tensorboard(
        self: AbstractModelHandler, *args, **kwargs
    ) -> bool:
        raise NotImplementedError(
            "Abstract method AbstractModelHandler::write_confidence_log_for_tensorboard"
            " should not be called."
        )

    def set_model(self: AbstractModelHandler, model) -> None:
        """
        Set the underlying model handled by this model handler.  This is generally
        called just once.
        """
        raise NotImplementedError(
            "Abstract method AbstractModelHandler::set_model should not be called."
        )

    def set_training_parameters(self: AbstractModelHandler) -> None:
        """
        Set the training parameters needed by the underlying model.  This is generally
        called just once and generally includes batch size, number of epochs, loss
        function, and stopping conditions.
        """
        raise NotImplementedError(
            "Abstract method AbstractModelHandler::set_training_parameters "
            "should not be called."
        )

    def train(
        self: AbstractModelHandler,
        train_features: NDArray,
        train_labels: NDArray,
        validation_features: NDArray = np.zeros(()),
        validation_labels: NDArray = np.zeros(()),
    ) -> None:
        """
        Ask the model to train.  This is generally called each time new labels have been
        provided.  Add training weights!!!
        """
        raise NotImplementedError(
            "Abstract method AbstractModelHandler::train should not be called."
        )

    def predict(self: AbstractModelHandler, features: NDArray, log_it: bool) -> NDArray:
        """
        Ask the model to make predictions.  This is generally called after training so
        that the strategy can show the user the new predictions and ask for corrections.
        Parameters include which examples should be predicted.
        """
        raise NotImplementedError(
            "Abstract method AbstractModelHandler::predict should not be called."
        )

    def compute_confidence_statistics(
        self: AbstractModelHandler, predictions: NDArray, percentiles: Sequence[float]
    ) -> Mapping:
        """
        Ask that the model provide statistics about its confidence in the supplied
        predictions.
        """
        raise NotImplementedError(
            "Abstract method AbstractModelHandler::compute_confidence_statistics "
            "should not be called."
        )


class GenericModelHandler(AbstractModelHandler):
    """
    GenericModelHandler handles functionality that is agnostic to the choice of the
    deep-learning framework.  GenericModelHandler is the superclass for
    TensorFlowModelHandler, PyTorchModelHandler, etc.  These subclasses handle
    AbstractModelHandler operations that are dependent upon which deep learning
    framework is being used.
    """

    def __init__(self) -> None:
        # super(GenericModelHandler, self).__init__()
        self.logger: Logger = Logger(self)

    def reset_log(self) -> None:
        self.logger.reset_log()

    def get_log(self) -> List:
        return self.get_logger().get_log()

    def get_logger(self) -> Logger:
        return self.logger

    def write_train_log_for_tensorboard(self, *args, **kwargs) -> bool:
        return self.logger.write_train_log_for_tensorboard(*args, **kwargs)

    def write_epoch_log_for_tensorboard(self, *args, **kwargs) -> bool:
        return self.logger.write_epoch_log_for_tensorboard(*args, **kwargs)

    def write_confidence_log_for_tensorboard(self, *args, **kwargs) -> bool:
        return self.logger.write_confidence_log_for_tensorboard(*args, **kwargs)

    def compute_confidence_statistics(
        self, predictions: NDArray, percentiles: Sequence[float]
    ) -> Mapping:
        # Compute several scores for each prediction.  High scores correspond to high
        # confidence.

        entropy_score: NDArray = -scipy.stats.entropy(predictions, axis=-1)
        margin_argsort: NDArray = np.argsort(predictions, axis=-1)
        prediction_indices: NDArray = np.arange(len(predictions))
        confidence_score: NDArray = predictions[
            prediction_indices, margin_argsort[:, -1]
        ]
        margin_score: NDArray = (
            confidence_score - predictions[prediction_indices, margin_argsort[:, -2]]
        )

        # Report percentile scores
        response: MutableMapping = dict()
        statistic_kind: str
        source_score: NDArray
        for statistic_kind, source_score in zip(
            ("maximum", "margin", "entropy"),
            (confidence_score, margin_score, entropy_score),
        ):
            response[statistic_kind] = dict()
            percentile_scores: NDArray = np.percentile(source_score, percentiles)
            for percentile, percentile_score in zip(percentiles, percentile_scores):
                response[statistic_kind][percentile] = percentile_score
        return response


class TensorFlowModelHandler(GenericModelHandler):
    """
    TensorFlowModelHandler is the class that implements framework-dependent
    GenericModelHandler routines via TensorFlow.
    """

    def __init__(self) -> None:
        super(TensorFlowModelHandler, self).__init__()
        self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False
        )

    def set_model(self, model) -> None:
        """
        Set the underlying model handled by this model handler.  This is generally
        called just once.
        """
        if not isinstance(model, tf.keras.Model):
            raise ValueError(
                "The parameter of TensorFlowModelHandler.set_model must be of type "
                "tf.keras.Model"
            )
        self.model = model

    def set_training_parameters(self) -> None:
        """
        Set the training parameters needed by the underlying model.  This is generally
        called just once and generally includes batch size, number of epochs, loss
        function, and stopping conditions.
        """
        raise NotImplementedError("Not implemented")

    def train(
        self,
        train_features: NDArray,
        train_labels: NDArray,
        validation_features: NDArray = np.zeros(()),
        validation_labels: NDArray = np.zeros(()),
    ) -> None:
        """
        Ask the model to train.  This is generally called each time new labels have been
        provided.  Add training weights!!!
        """
        assert not np.any(np.isnan(train_features))
        assert not np.any(np.isnan(train_labels))
        assert (len(validation_features) == 0) == (len(validation_labels) == 0)

        self.model.compile(
            optimizer="adam", loss=self.loss_function, metrics=["accuracy"]
        )

        validation_args: Dict = (
            dict()
            if len(validation_features) == 0
            else {"validation_data": (validation_features, validation_labels)}
        )
        # Get `epochs` from training parameters!!!
        self.logger.training_size = train_features.shape[0]
        self.model.fit(
            train_features,
            train_labels,
            epochs=10,
            verbose=0,
            callbacks=[self.logger],
            **validation_args,
        )
        # print(f"{repr(self.logger.get_log()) = }")

    def predict(self, features: NDArray, log_it: bool) -> NDArray:
        """
        Ask the model to make predictions.  This is generally called after training so
        that the strategy can show the user the new predictions and ask for corrections.
        Parameters include which examples should be predicted.
        """

        if log_it:
            predictions: NDArray = self.model.predict(
                features, verbose=0, callbacks=[self.logger]
            )
        else:
            predictions = self.model.predict(features, verbose=0)
        # print(f"{repr(self.logger.get_log()) = }")
        return predictions


class PyTorchModelHandler(GenericModelHandler):
    """
    PyTorchModelHandler is the class that implements framework-dependent
    GenericModelHandler routines via PyTorch.
    """

    def __init__(self) -> None:
        import torch

        super(PyTorchModelHandler, self).__init__()

        def categorical_cross_entropy(y_pred, y_true):
            y_pred = torch.clamp(y_pred, 1e-9, 1 - 1e-9)
            y_true = torch.eye(y_pred.shape[-1])[y_true]
            return -(y_true * torch.log(y_pred)).sum(dim=1).mean()

        self.criterion = categorical_cross_entropy

    def set_model(self, model) -> None:
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

    def set_training_parameters(self) -> None:
        """
        Set the training parameters needed by the underlying model.  This is generally
        called just once and generally includes batch size, number of epochs, loss
        function, and stopping conditions.
        """
        raise NotImplementedError("Not implemented")

    def train(
        self,
        train_features: NDArray,
        train_labels: NDArray,
        validation_features: NDArray = np.zeros(()),
        validation_labels: NDArray = np.zeros(()),
    ) -> None:
        """
        Ask the model to train.  This is generally called each time new labels have been
        provided.  Add training weights!!!
        """
        import torch

        assert not np.any(np.isnan(train_features))
        assert not np.any(np.isnan(train_labels))
        assert (len(validation_features) == 0) == (len(validation_labels) == 0)
        do_validation: bool = len(validation_features) != 0

        # This code heavily mimics
        # https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html.  For a
        # more detailed training loop example, see
        # https://towardsdatascience.com/a-tale-of-two-frameworks-985fa7fcec.

        self.logger.training_size = train_features.shape[0]
        self.logger.on_train_begin()

        # Get `epochs` from training parameters!!!
        number_of_epochs: int = 10
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

        class ZipDataset(torch.utils.data.Dataset):
            def __init__(self, train_features, train_labels) -> None:
                super(ZipDataset, self).__init__()
                self.train_features = torch.from_numpy(train_features)
                self.train_labels = torch.from_numpy(train_labels)

            def __len__(self):
                return self.train_labels.shape[0]

            def __getitem__(self, index: int):
                return self.train_features[index, :], self.train_labels[index]

        train_features_labels: ZipDataset = ZipDataset(train_features, train_labels)
        # Instead, get `batch_size` from somewhere!!!
        batch_size = 1
        # DataLoader has additional parameters that we may wish to use!!!
        my_train_data_loader = torch.utils.data.DataLoader(
            train_features_labels, batch_size=batch_size
        )

        if do_validation:
            validation_features_labels: ZipDataset = ZipDataset(
                validation_features, validation_labels
            )
            # DataLoader has additional parameters that we may wish to use!!!
            my_validation_data_loader = torch.utils.data.DataLoader(
                validation_features_labels, batch_size=batch_size
            )

        for epoch in range(number_of_epochs):  # loop over the dataset multiple times
            self.logger.on_epoch_begin(epoch)
            train_loss: float = 0.0
            train_size = 0
            train_correct = 0.0
            for i, data in enumerate(my_train_data_loader):
                self.logger.on_train_batch_begin(i)
                inputs, labels = data
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                new_size = inputs.size(0)
                train_size += new_size
                new_loss: float = loss.item() * inputs.size(0)
                train_loss += new_loss
                new_correct = (torch.argmax(outputs, dim=1) == labels).float().sum()
                train_correct += new_correct
                loss = new_loss / new_size
                accuracy = (new_correct / new_size).detach().cpu().numpy()
                if not isinstance(accuracy, (int, float, np.float32, np.float64)):
                    accuracy = accuracy[()]
                logs: Dict = {"loss": loss, "accuracy": accuracy}
                self.logger.on_train_batch_end(i, logs)
            loss = train_loss / train_size
            accuracy = (train_correct / train_size).detach().cpu().numpy()
            if not isinstance(accuracy, (int, float, np.float32, np.float64)):
                accuracy = accuracy[()]
            logs = {"loss": loss, "accuracy": accuracy}
            if do_validation:
                validation_loss: float = 0.0
                validation_size = 0
                validation_correct = 0.0
                for i, data in enumerate(my_validation_data_loader):
                    inputs, labels = data
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    new_size = inputs.size(0)
                    validation_size += new_size
                    new_loss = loss.item() * inputs.size(0)
                    validation_loss += new_loss
                    new_correct = (torch.argmax(outputs, dim=1) == labels).float().sum()
                    validation_correct += new_correct
                val_loss: float = validation_loss / validation_size
                val_accuracy = (
                    (validation_correct / validation_size).detach().cpu().numpy()[()]
                )
                more_logs: Dict = {"val_loss": val_loss, "val_accuracy": val_accuracy}
                logs = {**logs, **more_logs}
            self.logger.on_epoch_end(epoch, logs)

        self.logger.on_train_end(logs)  # `logs` is from the last epoch

    def predict(self, features: NDArray, log_it: bool) -> NDArray:
        """
        Ask the model to make predictions.  This is generally called after training so
        that the strategy can show the user the new predictions and ask for corrections.
        Parameters include which examples should be predicted.
        """
        import torch

        if log_it:
            self.logger.on_predict_begin()
        predictions_raw = self.model(torch.from_numpy(features))
        predictions: NDArray = predictions_raw.detach().cpu().numpy()
        if log_it:
            self.logger.on_predict_end({"outputs": predictions})
        return predictions


class AbstractEnsembleModelHandler(GenericModelHandler):
    """
    AbstractEnsembleModelHandler is a generic implementation of a GenericModelHandler
    that uses a committee of modelHandlers, via voting or similar, to determine its
    responses.  It's subclasses deliver an actual implementation.
    """

    def __init__(self) -> None:
        # super(AbstractEnsembleModelHandler, self).__init__()
        # raise NotImplementedError("Not implemented")
        pass

    def set_model(self, model) -> None:
        """
        Set the underlying model handled by this model handler.  This is generally
        called just once.
        """
        raise NotImplementedError("Not implemented")

    def set_training_parameters(self) -> None:
        """
        Set the training parameters needed by the underlying model.  This is generally
        called just once and generally includes batch size, number of epochs, loss
        function, and stopping conditions.
        """
        raise NotImplementedError("Not implemented")

    def train(
        self,
        train_features: NDArray,
        train_labels: NDArray,
        validation_features: NDArray = np.zeros(()),
        validation_labels: NDArray = np.zeros(()),
    ):
        """
        Ask the model to train.  This is generally called each time new labels have been
        provided.  Add training weights!!!
        """
        raise NotImplementedError("Not implemented")

    def predict(self, features: NDArray, log_it: bool):
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

    def __init__(self) -> None:
        super(ExampleEnsembleModelHandler, self).__init__()
        # raise NotImplementedError("Not implemented")

    def set_model(self, model) -> None:
        """
        Set the underlying model handled by this model handler.  This is generally
        called just once.
        """
        raise NotImplementedError("Not implemented")

    def set_training_parameters(self) -> None:
        """
        Set the training parameters needed by the underlying model.  This is generally
        called just once and generally includes batch size, number of epochs, loss
        function, and stopping conditions.
        """
        raise NotImplementedError("Not implemented")

    def train(
        self,
        train_features: NDArray,
        train_labels: NDArray,
        validation_features: NDArray = np.zeros(()),
        validation_labels: NDArray = np.zeros(()),
    ) -> None:
        """
        Ask the model to train.  This is generally called each time new labels have been
        provided.  Add training weights!!!
        """
        raise NotImplementedError("Not implemented")

    def predict(self, features: NDArray, log_it: bool) -> NDArray:
        """
        Ask the model to make predictions.  This is generally called after training so
        that the strategy can show the user the new predictions and ask for corrections.
        Parameters include which examples should be predicted.
        """
        raise NotImplementedError("Not implemented")
