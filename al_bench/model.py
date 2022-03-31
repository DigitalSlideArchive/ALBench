import numpy as np


class GenericModel:
    """
    Functionality that is agnostic to the choice of the deep-learning framework is
    handled by this class.  GenericModel is the superclass for TensorFlowModel, PyTorchModel,
    etc.  These subclasses handle GenericModel operations that are dependent upon which deep
    learning framework is being used.
    """

    def __init__():
        pass

    def set_model(model):
        raise ValueError(
            "Abstract method GenericModel::set_model should not be called."
        )

    def set_desired_outputs():
        # Write me!!!  Alternatively, record everything and then have a method that
        # selectively fetches some of them.  This may be called from subclass overrides.
        pass

    def set_training_parameters():
        # Write me!!!  Generally called just once.  Includes batch size, number of
        # epochs, loss function / stopping conditions.
        pass

    def set_all_labels():
        # Write me!!!  Be able to handle different treatments for newly labeled
        # examples; generally, handle a weighting scheme that depends upon when examples
        # were labeled.
        pass

    def set_some_labels():
        # Write me!!!  Be able to handle different treatments for newly labeled
        # examples; generally, handle a weighting scheme that depends upon when examples
        # were labeled.
        pass

    def train():
        # Write me!!!  Generally called each time we train.  Includes dataset; should
        # this already be divided into train, test, verify?  Should it already have
        # weights?
        pass

    def predict():
        # Write me!!! Which members of the data set should we predict for?
        pass


class TensorFlowModel(GenericModel):
    """
    TensorFlowModel is the class that implements framework-agnostic GenericModel routines via
    TensorFlow.
    """

    def __init__():
        pass


class PyTorchModel(GenericModel):
    """
    PyTorchModel is the class that implements framework-agnostic GenericModel routines via
    PyTorch.
    """

    def __init__():
        pass


class GenericEnsembleModel(GenericModel):
    """
    GenericEnsembleModel is a generic implementation of a GenericModel that uses a
    committee of models, via voting or similar, to determine its responses.  It's
    subclasses deliver an actual implementation.
    """

    def __init__():
        pass


class ExampleEnsembleModel(GenericEnsembleModel):
    """
    ExampleEnsembleModel is a specific implementation of a GenericEnsembleModel that
    uses a committee of models, via voting or similar, to determine its responses.  The
    methods of ExampleEnsembleModel determine the specific implementation of this
    ensemble.
    """

    def __init__():
        pass
