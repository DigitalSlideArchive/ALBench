"""
# ############
# INSTALLATION
# ############

# Create, launch, and update a Python virtual environment as ~/venv/superpixel or
# elsewhere.
python -m venv ~/venv/superpixel
source ~/venv/superpixel/bin/activate
pip install -U pip setuptools wheel

# Install h5py and tensorflow
pip install h5py tensorflow

# Note: We also need to have the module networks.py in the same directory as the present
# executable file, Predict.py.

# ############
# RUN
# ############

# Use a command like:
python Predict.py features.h5 labels.h5 predict.h5 BRCA-spfeatures-1 predictions.h5
# where features.h5, labels.h5 and predict.h5 are input files and
# checkpoints/BRCA-spfeatures-1.h5 and predictions.h5 are where the
# output is written.

python Predict.py -h
# produces:
#   usage: Predict.py [-h] features labels predict classifier predictions
#
#   Make some predictions.
#
#   positional arguments:
#     features     h5py file containing a "features" (S, F)-array of the F feature
#                  values for S superpixels
#     labels       h5py file containing a "labels" (L, 2)-array, which is the index
#                  (into "features") and label for each of L labeled superpixels
#     predict      h5py file containing a "predict" (P,)-array, which is the index (into
#                  "features") for each of P superpixels to make predictions for
#     classifier   ./checkpoints/classifier.h5 is the h5py file to write the model to
#     predictions  h5py file to write out, containing a "predictions" (P, 2)-array,
#                  which is the index (into "features") and label for each of P
#                  predicted superpixels
#
#   optional arguments:
#     -h, --help   show this help message and exit
"""

import argparse

import h5py as h5
import numpy as np

import networks


def main():
    # Create parser
    #
    parser = argparse.ArgumentParser(description="Make some predictions.")
    parser.add_argument(
        "features",
        metavar="features",
        type=str,
        nargs=1,
        help='h5py file containing a "features" (S, F)-array of the F feature values '
        "for S superpixels",
    )
    parser.add_argument(
        "labels",
        metavar="labels",
        type=str,
        nargs=1,
        help='h5py file containing a "labels" (L, 2)-array, which is the index (into '
        '"features") and label for each of L labeled superpixels',
    )
    parser.add_argument(
        "predict",
        metavar="predict",
        type=str,
        nargs=1,
        help='h5py file containing a "predict" (P,)-array, which is the index (into '
        '"features") for each of P superpixels to make predictions for',
    )
    parser.add_argument(
        "classifier",
        metavar="classifier",
        type=str,
        nargs=1,
        help="./checkpoints/classifier.h5 is the h5py file to write the model to",
    )
    parser.add_argument(
        "predictions",
        metavar="predictions",
        type=str,
        nargs=1,
        help='h5py file to write out, containing a "predictions" (P, 2)-array, which '
        'is the index (into "features") and label for each of P predicted superpixels',
    )

    # Use parser to load inputs
    #
    args = parser.parse_args()

    classifier = args.classifier[0]

    # Read features from disk.  features = np.array([(64-term tuple), (64-term tuple),
    # ...])
    with h5.File(args.features[0]) as ds:
        features = np.array(ds["features"])
        if False:
            # If `features` is an np.array of shape (N, 64) then it can be written to
            # file with:
            with h5.File("features.h5", "w") as f:
                f.create_dataset("features", features.shape, data=features)

    # Read labels from disk.  labels = np.array([(index, label) for index, label in
    # labels])
    with h5.File(args.labels[0]) as ds:
        labels = np.array(ds["labels"])
        if False:
            # If `labels` is an np.array of shape (N, 2) then it can be written to file
            # with:
            with h5.File("labels.h5", "w") as f:
                f.create_dataset("labels", labels.shape, data=labels)

    # Read `predict` from disk.  predict = np.array([index, index, ...])
    with h5.File(args.predict[0]) as ds:
        predict = np.array(ds["predict"])
        if False:
            # If `predict` is an np.array of shape (N,) then it can be written to file
            # with:
            with h5.File("predict.h5", "w") as f:
                f.create_dataset("predict", predict.shape, data=predict)

    # Build and train the model
    model = networks.Network()
    model.init_model()
    # classifier = "BRCA-spfeatures-1"
    # Model we be saved as ./checkpoints/classifier.h5
    model.train_model(features[labels[:, 0], :], labels[:, 1], classifier)

    # Compute the predictions
    column_shape = predict.shape + (1,)
    predictions = np.hstack(
        (
            predict.reshape(column_shape),
            model.predict_classes(features[predict, :]).reshape(column_shape),
        )
    )

    # Write out the predictions
    with h5.File(args.predictions[0], "w") as f:
        f.create_dataset("predictions", predictions.shape, data=predictions)


if __name__ == "__main__":
    main()
