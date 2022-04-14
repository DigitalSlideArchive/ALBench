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

import h5py as h5
import numpy as np


class DatasetHandler:
    def __init__(self):
        self.features = None
        self.labels = None
        self.dictionaries = None
        self.label_definitions = None

    """
    Handle the vector of features for each stored entity.
    """

    def read_all_features_from_h5py(self, filename, data_name="features"):
        with h5.File(filename) as ds:
            self.features = np.array(ds[data_name])

    def write_all_features_to_h5py(self, filename, data_name="features"):
        with h5.File(filename, "w") as f:
            f.create_dataset(data_name, self.features.shape, data=self.features)

    def set_all_features(self, features):
        if isinstance(features, np.ndarray) and len(features.shape) == 2:
            self.features = features
        else:
            raise ValueError(
                "The argument to set_all_features must be a 2-dimensional numpy ndarray."
            )

    def get_all_features(self):
        return self.features

    def clear_all_features(self):
        self.features = None

    def set_some_features(self, feature_indices, features):
        """
        This overwrites existing features.

        N.B. as with numpy arrays in general, a single feature index and a list with one
        feature index will have different behaviors, in that the former drops an array
        dimension but the latter does not.
        """

        # This does not (yet) handle insert, delete, or append.
        self.features[feature_indices,] = features

    def get_some_features(self, feature_indices):
        """
        N.B. as with numpy arrays in general, a single feature index and a list with one
        feature index will have different behaviors, in that the former drops an array
        dimension but the latter does not.
        """
        return self.features[feature_indices,]

    """
    Handle the label(s) for each stored entity.
    """

    def read_all_labels_from_h5py(self, filename, data_name="labels"):
        with h5.File(filename) as ds:
            self.labels = np.array(ds[data_name])

    def write_all_labels_to_h5py(self, filename, data_name="labels"):
        with h5.File(filename, "w") as f:
            f.create_dataset(data_name, self.labels.shape, data=self.labels)

    def set_all_labels(self, labels):
        if isinstance(labels, np.ndarray) and 1 <= len(labels.shape) <= 2:
            self.labels = labels
        else:
            raise ValueError(
                "The argument to set_all_labels must be a 1-dimensional or 2-dimensional "
                "numpy ndarray."
            )

    def get_all_labels(self):
        return self.labels

    def clear_all_labels(self):
        self.labels = None

    def set_some_labels(self, label_indices, labels):
        """
        This overwrites existing labels.

        N.B. as with numpy arrays in general, a single label index and a list with one
        label index will have different behaviors, in that the former drops an array
        dimension but the latter does not.

        This does not (yet) handle insert, delete, or append.
        """

        self.labels[label_indices,] = labels

    def get_some_labels(self, label_indices):
        """
        N.B. as with numpy arrays in general, a single label index and a list with one
        label index will have different behaviors, in that the former drops an array
        dimension but the latter does not.
        """
        return self.labels[label_indices,]

    """
    Handle the dictionary of supplemental information for each stored entity.
    """

    def set_all_dictionaries(self, dictionaries):
        """
        N.B. if we were handed a (read-only) tuple then we cannot change a subset of
        them.  If this functionality is needed then supply `list(dictionaries)` to this
        function.
        """
        if (
            isinstance(dictionaries, (tuple, list))
            and len(dictionaries) > 0
            and isinstance(dictionaries[0], dict)
        ):
            self.dictionaries = dictionaries
        else:
            raise ValueError(
                "The argument to set_all_dictionaries must be a non-empty list or tuple "
                "of dictionaries."
            )

    def get_all_dictionaries(self):
        return self.dictionaries

    def clear_all_dictionaries(self):
        self.dictionaries = None

    def set_some_dictionaries(self, dictionary_indices, dictionaries):
        """
        This overwrites existing dictionaries.

        N.B. as with numpy arrays in general, a single dictionary index and a list with
        one dictionary index will have different behaviors, in that the former drops an
        array dimension but the latter does not.
        """

        # This will fail if the intially supplied value for dictionaries was a *tuple*
        # of dictionaries rather than a *list* of dictionaries, because tuples are
        # read-only.
        if isinstance(self.dictionaries, tuple):
            raise ValueError(
                "set_some_dictionaries cannot be used unless the initially supplied "
                "dictionaries are supplied in a list instead of a tuple"
            )

        # This does not (yet) handle insert, delete, or append.
        self.dictionaries[dictionary_indices,] = dictionaries

    def get_some_dictionaries(self, dictionary_indices):
        """
        N.B. as with numpy arrays in general, a single dictionary index and a list with
        one dictionary index will have different behaviors, in that the former drops an
        array dimension but the latter does not.
        """
        return self.dictionaries[dictionary_indices,]

    """
    Handle operations not specific to the feature vectors, labels, or dictionaries of
    supplemental information of each stored entity.
    """

    def set_all_label_definitions(self, label_definitions):
        """
        The argument is, for example, label_definitions = {
            np.nan: {"description": "unlabeled", "color": "#FFFF00"},
            1:      {"description": "necrotic",  "color": "#FF0000"},
        }

        """

        if isinstance(label_definitions, dict):
            self.label_definitions = label_definitions

    def get_all_label_definitions(self):
        return self.label_definitions

    def check_data_consistency(self):
        """
        Among features, labels, and dictionaries that were supplied, are they for the
        same number of entities?
        """
        features_length = 0 if self.features is None else self.features.shape[0]
        labels_length = 0 if self.labels is None else self.labels.shape[0]
        dictionaries_length = 0 if self.dictionaries is None else len(self.dictionaries)
        all_lengths = set([features_length, labels_length, dictionaries_length])
        lengths_test = len(all_lengths) == 1 or (
            len(all_lengths) == 2 and 0 in all_lengths
        )
        # Print output if lengths_test fails!!!
        """
        Check that the label_definitions include at least all the labels that have been
        used!!!
        """
        definitions_test = True
        # Print output if definitions_test fails!!!

        return lengths_test and definitions_test
