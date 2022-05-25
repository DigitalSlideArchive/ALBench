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

import h5py as h5
import numpy as np


class AbstractDatasetHandler:
    """
    Abstract base class for dataset handlers.
    """

    def __init__(self):
        raise NotImplementedError(
            "Abstract method AbstractDatasetHandler::__init__ should not be called."
        )

    def read_all_features_from_h5py(self, filename, data_name="features"):
        raise NotImplementedError(
            "Abstract method AbstractDatasetHandler::read_all_features_from_h5py "
            "should not be called."
        )

    def write_all_features_to_h5py(self, filename, data_name="features"):
        raise NotImplementedError(
            "Abstract method AbstractDatasetHandler::write_all_features_to_h5py "
            "should not be called."
        )

    def set_all_features(self, features):
        raise NotImplementedError(
            "Abstract method AbstractDatasetHandler::set_all_features "
            "should not be called."
        )

    def get_all_features(self):
        raise NotImplementedError(
            "Abstract method AbstractDatasetHandler::get_all_features "
            "should not be called."
        )

    def clear_all_features(self):
        raise NotImplementedError(
            "Abstract method AbstractDatasetHandler::clear_all_features "
            "should not be called."
        )

    def set_some_features(self, feature_indices, features):
        raise NotImplementedError(
            "Abstract method AbstractDatasetHandler::set_some_features "
            "should not be called."
        )

    def get_some_features(self, feature_indices):
        raise NotImplementedError(
            "Abstract method AbstractDatasetHandler::get_some_features "
            "should not be called."
        )

    def read_all_labels_from_h5py(self, filename, data_name="labels"):
        raise NotImplementedError(
            "Abstract method AbstractDatasetHandler::read_all_labels_from_h5py "
            "should not be called."
        )

    def write_all_labels_to_h5py(self, filename, data_name="labels"):
        raise NotImplementedError(
            "Abstract method AbstractDatasetHandler::write_all_labels_to_h5py "
            "should not be called."
        )

    def set_all_labels(self, labels):
        raise NotImplementedError(
            "Abstract method AbstractDatasetHandler::set_all_labels "
            "should not be called."
        )

    def get_all_labels(self):
        raise NotImplementedError(
            "Abstract method AbstractDatasetHandler::get_all_labels "
            "should not be called."
        )

    def clear_all_labels(self):
        raise NotImplementedError(
            "Abstract method AbstractDatasetHandler::clear_all_labels "
            "should not be called."
        )

    def set_some_labels(self, label_indices, labels):
        raise NotImplementedError(
            "Abstract method AbstractDatasetHandler::set_some_labels "
            "should not be called."
        )

    def get_some_labels(self, label_indices):
        raise NotImplementedError(
            "Abstract method AbstractDatasetHandler::get_some_labels "
            "should not be called."
        )

    def set_all_dictionaries(self, dictionaries):
        raise NotImplementedError(
            "Abstract method AbstractDatasetHandler::set_all_dictionaries "
            "should not be called."
        )

    def get_all_dictionaries(self):
        raise NotImplementedError(
            "Abstract method AbstractDatasetHandler::get_all_dictionaries "
            "should not be called."
        )

    def clear_all_dictionaries(self):
        raise NotImplementedError(
            "Abstract method AbstractDatasetHandler::clear_all_dictionaries "
            "should not be called."
        )

    def set_some_dictionaries(self, dictionary_indices, dictionaries):
        raise NotImplementedError(
            "Abstract method AbstractDatasetHandler::set_some_dictionaries "
            "should not be called."
        )

    def get_some_dictionaries(self, dictionary_indices):
        raise NotImplementedError(
            "Abstract method AbstractDatasetHandler::get_some_dictionaries "
            "should not be called."
        )

    def set_all_label_definitions(self, label_definitions):
        raise NotImplementedError(
            "Abstract method AbstractDatasetHandler::set_all_label_definitions "
            "should not be called."
        )

    def get_all_label_definitions(self):
        raise NotImplementedError(
            "Abstract method AbstractDatasetHandler::get_all_label_definitions "
            "should not be called."
        )

    def check_data_consistency(self):
        raise NotImplementedError(
            "Abstract method AbstractDatasetHandler::check_data_consistency "
            "should not be called."
        )


class BasicDatasetHandler(AbstractDatasetHandler):
    def __init__(self):
        self.features = None
        self.labels = None
        self.dictionaries = None
        self.label_definitions = None

    """
    Handle the vector of features for each stored entity.
    """

    def read_all_features_from_h5py(self, filename, data_name="features"):
        """
        Read the entire database of features from a supplied h5py file.
        """
        with h5.File(filename) as ds:
            self.features = np.array(ds[data_name])

    def write_all_features_to_h5py(self, filename, data_name="features"):
        """
        Write the entire database of features to a h5py file.
        """
        with h5.File(filename, "w") as f:
            f.create_dataset(data_name, self.features.shape, data=self.features)

    def set_all_features(self, features):
        """
        Set the entire database of features from a supplied numpy array.
        """
        if isinstance(features, np.ndarray) and len(features.shape) == 2:
            self.features = features
        else:
            raise ValueError(
                "The argument to set_all_features must be a 2-dimensional numpy "
                "ndarray."
            )

    def get_all_features(self):
        """
        Get the entire database of features as a numpy array.
        """
        return self.features

    def clear_all_features(self):
        """
        Remove all features from the database
        """
        self.features = None

    def set_some_features(self, feature_indices, features):
        """
        This overwrites existing features.  It does not (yet) handle insert, delete, or
        append.

        N.B. as with numpy arrays in general, a single feature index and a list with one
        feature index will have different behaviors, in that the former drops an array
        dimension but the latter does not.  That is, when changing one feature use
        feature_indices=5, features=[3,1.2,4] OR use feature_indices=[5],
        features=[[3,1.2,4]].
        """
        self.features[feature_indices] = features

    def get_some_features(self, feature_indices):
        """
        This fetches a subset of existing features.

        N.B. as with numpy arrays in general, a single feature index and a list with one
        feature index will have different behaviors, in that the former drops an array
        dimension but the latter does not.  That is, use of feature_indices=5 returns
        [3,1.2,4] but use of feature_indices=[5] returns [[3,1.2,4]].
        """
        return self.features[feature_indices]

    """
    Handle the label(s) for each stored entity.
    """

    def read_all_labels_from_h5py(self, filename, data_name="labels"):
        """
        Read the entire database of labels from a supplied h5py file.
        """
        with h5.File(filename) as ds:
            self.labels = np.array(ds[data_name])

    def write_all_labels_to_h5py(self, filename, data_name="labels"):
        """
        Write the entire database of labels to a h5py file.
        """
        with h5.File(filename, "w") as f:
            f.create_dataset(data_name, self.labels.shape, data=self.labels)

    def set_all_labels(self, labels):
        """
        Set the entire database of labels from a supplied numpy array.
        """
        if isinstance(labels, np.ndarray) and 1 <= len(labels.shape) <= 2:
            self.labels = labels
        else:
            raise ValueError(
                "The argument to set_all_labels must be a 1-dimensional or "
                "2-dimensional numpy ndarray."
            )

    def get_all_labels(self):
        """
        Get the entire database of labels as a numpy array.
        """
        return self.labels

    def clear_all_labels(self):
        """
        Remove all labels from the database
        """
        self.labels = None

    def set_some_labels(self, label_indices, labels):
        """
        This overwrites existing labels.  It does not (yet) handle insert, delete, or
        append.

        N.B. as with numpy arrays in general, a single label index and a list with one
        label index will have different behaviors, in that the former drops an array
        dimension but the latter does not.  That is, when changing one label use
        label_indices=5, labels=[3,1,4] OR use label_indices=[5],
        labels=[[3,1,4]].
        """
        self.labels[label_indices] = labels

    def get_some_labels(self, label_indices):
        """
        This fetches a subset of existing labels.

        N.B. as with numpy arrays in general, a single label index and a list with one
        label index will have different behaviors, in that the former drops an array
        dimension but the latter does not.  That is, use of label_indices=5 returns
        [3,1,4] but use of label_indices=[5] returns [[3,1,4]].
        """
        return self.labels[
            label_indices,
        ]

    """
    Handle the dictionary of supplemental information for each stored entity.
    """

    def set_all_dictionaries(self, dictionaries):
        """
        Set the entire database of dictionaries -- one per feature vector -- from a
        supplied list of Python dict objects.

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
                "The argument to set_all_dictionaries must be a non-empty list or "
                "tuple of dictionaries."
            )

    def get_all_dictionaries(self):
        """
        Get the entire database of dictionaries as a list (or tuple) of Python dict
        objects.
        """
        return self.dictionaries

    def clear_all_dictionaries(self):
        """
        Remove all dictionaries from the database
        """
        self.dictionaries = None

    def set_some_dictionaries(self, dictionary_indices, dictionaries):
        """
        This overwrites existing dictionaries.  It does not (yet) handle insert, delete,
        or append.

        N.B. as with Python lists in general, a single dictionary index and a list with
        one dictionary index will have different behaviors, in that the former drops an
        array dimension but the latter does not.  That is, when changing one dictionary
        use dictionary_indices=5, dictionaries={'a': 1, 'b': 2} OR use
        dictionary_indices=[5], dictionaries=[{'a': 1, 'b': 2}].

        N.B. This will fail if the intially supplied value for dictionaries was a
        *tuple* of dictionaries rather than a *list* of dictionaries, because tuples are
        read-only.
        """
        if isinstance(self.dictionaries, tuple):
            raise ValueError(
                "set_some_dictionaries cannot be used unless the initially supplied "
                "dictionaries are supplied in a list instead of a tuple"
            )

        self.dictionaries[dictionary_indices] = dictionaries

    def get_some_dictionaries(self, dictionary_indices):
        """
        This fetches a subset of existing dictionaries.

        N.B. as with numpy arrays in general, a single dictionary index and a list with
        one dictionary index will have different behaviors, in that the former drops an
        array dimension but the latter does not.  That is, use of dictionary_indices=5
        returns {'a': 1, 'b': 2} but use of dictionary_indices=[5] returns [{'a': 1,
        'b': 2}].
        """
        return self.dictionaries[dictionary_indices]

    """
    Handle operations not specific to the feature vectors, labels, or dictionaries of
    supplemental information of each stored entity.
    """

    def set_all_label_definitions(self, label_definitions):
        """
        Parameters
        ----------
        label_definitions: dict
            The argument is, for example, label_definitions = {
                np.nan: {"description": "unlabeled", "color": "#FFFF00"},
                1:      {"description": "necrotic",  "color": "#FF0000"},
            }
        """

        if isinstance(label_definitions, list) and all(
            [isinstance(e, dict) for e in label_definitions]
        ):
            self.label_definitions = label_definitions
        else:
            raise ValueError(
                "The argument to set_all_label_definitions must be a list of Python "
                "dict objects"
            )

    def get_all_label_definitions(self):
        """
        Returns
        -------
        label_definitions: dict
            See set_all_label_definitions for the dict format.
        """
        return self.label_definitions

    def check_data_consistency(self):
        # Check whether among features, labels, and dictionaries that were supplied, are
        # they for the same number of entities?
        features_length = 0 if self.features is None else self.features.shape[0]
        labels_length = 0 if self.labels is None else self.labels.shape[0]
        dictionaries_length = 0 if self.dictionaries is None else len(self.dictionaries)
        # Eliminate duplicates
        all_lengths = set([features_length, labels_length, dictionaries_length])
        lengths_test = len(all_lengths) == 1 or (
            len(all_lengths) == 2 and 0 in all_lengths
        )

        # Check whether among labels and label_definitions that were supplied, are
        # they for the same number of kinds of labels?
        labels_width = (
            0
            if self.labels is None
            else 1
            if len(self.labels.shape) == 1
            else self.labels.shape[1]
        )
        label_definitions_width = (
            0 if self.label_definitions is None else len(self.label_definitions)
        )
        all_widths = set([labels_width, label_definitions_width])
        widths_test = len(all_widths) == 1 or (len(all_widths) == 2 and 0 in all_widths)

        ## Check whether every supplied label category has a definition.
        definitions_test = (
            not widths_test  # already failed previous test
            or labels_width == 0  # nothing to compare
            or label_definitions_width == 0  # nothing to compare
            or (
                len(self.labels.shape) == 1  # 1-dimensional numpy array
                and len(
                    set(self.labels).difference(set(self.label_definitions[0].keys()))
                )
                == 0
            )
            or (
                len(self.labels.shape) == 2  # 2-dimensional numpy array
                and all(
                    [
                        len(
                            set(self.labels[:, col]).difference(
                                set(self.label_definitions[col].keys())
                            )
                        )
                        == 0
                        for col in range(self.labels.shape[1])
                    ]
                )
            )
        )

        mesgs = list()
        if not lengths_test:
            mesgs += [
                f"height(features) = {features_length}, "
                f"height(labels) = {labels_length}, and "
                f"height(dictionaries) = {dictionaries_length} do not match."
            ]
        if not widths_test:
            mesgs += [
                f"width(labels) = {labels_width} and "
                f"width(label_definitions) = {label_definitions_width} do not match."
            ]
        if not definitions_test:
            mesgs += [f"Some labels have categories without definitions."]
        if len(mesgs) > 0:
            raise ValueError("\n".join(mesgs))

        return len(mesgs) == 0
