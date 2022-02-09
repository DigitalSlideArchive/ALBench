import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import (
    Adam,
    RMSprop,
    Adadelta,
    SGD,
    Adagrad,
    Adamax,
    Nadam,
)

"""
Network class

Initialize neural network model.
Perform train model.
Return predicted probabilities/predicted labels.
"""


class Network:
    def __init__(self):
        # set variables
        self.input_units = 64
        self.hidden_units = 32
        self.output_units = 1
        self.epochs = 20
        self.dropout = 0.3
        self.activation = "relu"
        self.activation_last = "sigmoid"
        self.optimizer = "Adam"
        self.learning_rate = 0.001
        self.loss = "binary_crossentropy"
        self.noise_shape = None
        self.seed = 145
        self.batch_size = 1000000
        self.metrics = "accuracy"
        self.checkpointIter = 0
        self.model = None
        self.classifier = None

    """
    Network.set_params requires that exactly five parameters be supplied, as a dict or list.
    """

    def set_params(self, q):
        if isinstance(q, dict) and len(q) == 5:
            self.activation = str(q["activation"])
            self.optimizer = str(q["optimizer"])
            self.epochs = int(q["epochs"])
            self.learning_rate = float(q["learning_rate"])
            self.dropout = float(q["dropout"])
        elif isinstance(q, list) and len(q) == 5:
            self.activation = str(q[0])
            self.optimizer = str(q[1])
            self.epochs = int(q[2])
            self.learning_rate = float(q[3])
            self.dropout = float(q[4])
        else:
            print(f"Network.set_params({q}) failed")

    def get_params(self):
        data = {}
        data["activation"] = self.activation
        data["optimizer"] = self.optimizer
        data["epochs"] = str(self.epochs)
        data["learning_rate"] = str(self.learning_rate)
        data["dropout"] = str(self.dropout)
        return data

    def init_model(self):
        self.model = tf.keras.models.Sequental()
        self.model.add(
            tf.keras.layers.Dense(
                self.hidden_units,
                input_dim=self.input_units,
                activation=self.activation,
            )
        )
        self.model.add(
            tf.keras.layers.Dropout(
                self.dropout, noise_shape=self.noise_shape, seed=self.seed
            )
        )
        self.model.add(
            tf.keras.layers.Dense(self.output_units, activation=self.activation_last)
        )
        if self.optimizer == "RMSprop":
            opt = tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate)
        elif self.optimizer == "Adadelta":
            opt = tf.keras.optimizers.Adadelta(learning_rate=self.learning_rate)
        elif self.optimizer == "SGD":
            opt = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        elif self.optimizer == "Adagrad":
            opt = tf.keras.optimizers.Adagrad(learning_rate=self.learning_rate)
        elif self.optimizer == "Adamax":
            opt = tf.keras.optimizers.Adamax(learning_rate=self.learning_rate)
        elif self.optimizer == "Nadam":
            opt = tf.keras.optimizers.Nadam(learning_rate=self.learning_rate)
        else:
            opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
            self.model.compile(optimizer=opt, loss=self.loss, metrics=[self.metrics])

    def loading_model(self, path):
        self.model = tf.keras.models.load_model(path)

    def saving_model(self, path):
        self.model.save(path)

    def train_model(self, features, labels, classifier):
        self.classifier = classifier
        self.model.fit(features, labels, epochs=self.epochs)
        self.model.save_weights("./checkpoints/" + self.classifier + ".h5")

    def predict(self, features):
        if self.classifier:
            self.model.load_weights("./checkpoints/" + self.classifier + ".h5")
            predicts = self.model.predict(features, batch_size=self.batch_size)[:, 0]
            return predicts

    def predict_classes(self, features):
        if self.classifier:
            if self.activation_last == "sigmoid":
                # binary classification via 'sigmoid' as last-layer activation
                predicts_classes = (self.predict(features) > 0.5).astype("int32")
            else:
                # multi-class classification via 'softmax' as last-layer activation
                predicts_classes = np.argmax(self.predict(features), axis=-1)
            return predicts_classes


def run():
    model = Network()
    model.init_model()
