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
import h5py as h5
import numpy as np
from numpy.typing import NDArray
from typing import Iterable, List, MutableMapping


class AbstractDatasetHandler:
    """
    Abstract base class for dataset handlers.
    """

    def __init__(self) -> None:
        raise NotImplementedError(
            "Abstract method AbstractDatasetHandler::__init__ should not be called."
        )

    def read_all_feature_vectors_from_h5py(
        self, filename: str, data_name: str = "features"
    ) -> None:
        raise NotImplementedError(
            "Abstract method "
            "AbstractDatasetHandler::read_all_feature_vectors_from_h5py should not be "
            "called."
        )

    def write_all_feature_vectors_to_h5py(
        self, filename: str, data_name: str = "features"
    ) -> None:
        raise NotImplementedError(
            "Abstract method AbstractDatasetHandler::write_all_feature_vectors_to_h5py "
            "should not be called."
        )

    def set_all_feature_vectors(self, feature_vectors: NDArray) -> None:
        raise NotImplementedError(
            "Abstract method AbstractDatasetHandler::set_all_feature_vectors "
            "should not be called."
        )

    def get_all_feature_vectors(self) -> NDArray:
        raise NotImplementedError(
            "Abstract method AbstractDatasetHandler::get_all_feature_vectors "
            "should not be called."
        )

    def clear_all_feature_vectors(self) -> None:
        raise NotImplementedError(
            "Abstract method AbstractDatasetHandler::clear_all_feature_vectors "
            "should not be called."
        )

    def set_some_feature_vectors(
        self, feature_vector_indices: NDArray, feature_vectors: NDArray
    ) -> None:
        raise NotImplementedError(
            "Abstract method AbstractDatasetHandler::set_some_feature_vectors "
            "should not be called."
        )

    def get_some_feature_vectors(self, feature_vector_indices: NDArray) -> NDArray:
        raise NotImplementedError(
            "Abstract method AbstractDatasetHandler::get_some_feature_vectors "
            "should not be called."
        )

    def get_training_feature_vectors(self) -> NDArray:
        raise NotImplementedError(
            "Abstract method AbstractDatasetHandler::get_training_feature_vectors "
            "should not be called."
        )

    def get_validation_feature_vectors(self) -> NDArray:
        raise NotImplementedError(
            "Abstract method AbstractDatasetHandler::get_validation_feature_vectors "
            "should not be called."
        )

    def read_all_labels_from_h5py(
        self, filename: str, data_name: str = "labels"
    ) -> None:
        raise NotImplementedError(
            "Abstract method AbstractDatasetHandler::read_all_labels_from_h5py "
            "should not be called."
        )

    def write_all_labels_to_h5py(
        self, filename: str, data_name: str = "labels"
    ) -> None:
        raise NotImplementedError(
            "Abstract method AbstractDatasetHandler::write_all_labels_to_h5py "
            "should not be called."
        )

    def set_all_labels(self, labels: NDArray) -> None:
        raise NotImplementedError(
            "Abstract method AbstractDatasetHandler::set_all_labels "
            "should not be called."
        )

    def get_all_labels(self) -> NDArray:
        raise NotImplementedError(
            "Abstract method AbstractDatasetHandler::get_all_labels "
            "should not be called."
        )

    def clear_all_labels(self) -> None:
        raise NotImplementedError(
            "Abstract method AbstractDatasetHandler::clear_all_labels "
            "should not be called."
        )

    def set_some_labels(self, label_indices: NDArray, labels: NDArray) -> None:
        raise NotImplementedError(
            "Abstract method AbstractDatasetHandler::set_some_labels "
            "should not be called."
        )

    def get_some_labels(self, label_indices: NDArray) -> NDArray:
        raise NotImplementedError(
            "Abstract method AbstractDatasetHandler::get_some_labels "
            "should not be called."
        )

    def get_training_labels(self) -> NDArray:
        raise NotImplementedError(
            "Abstract method AbstractDatasetHandler::get_training_labels "
            "should not be called."
        )

    def get_validation_labels(self) -> NDArray:
        raise NotImplementedError(
            "Abstract method AbstractDatasetHandler::get_validation_labels "
            "should not be called."
        )

    def query_oracle(self, next_indices: NDArray) -> NDArray:
        """
        This method queries the oracle for labels for the supplied indices.  It returns
        all labels.  Note that in a simulation, we already have those labels and there
        is nothing to do.
        """
        raise NotImplementedError(
            "Abstract method AbstractDatasetHandler::query_oracle should not be called."
        )

    def set_all_dictionaries(self, dictionaries: Iterable[MutableMapping]) -> None:
        raise NotImplementedError(
            "Abstract method AbstractDatasetHandler::set_all_dictionaries "
            "should not be called."
        )

    def get_all_dictionaries(self) -> Iterable[MutableMapping]:
        raise NotImplementedError(
            "Abstract method AbstractDatasetHandler::get_all_dictionaries "
            "should not be called."
        )

    def clear_all_dictionaries(self) -> None:
        raise NotImplementedError(
            "Abstract method AbstractDatasetHandler::clear_all_dictionaries "
            "should not be called."
        )

    def set_some_dictionaries(
        self, dictionary_indices: NDArray, dictionaries: Iterable[MutableMapping]
    ) -> None:
        raise NotImplementedError(
            "Abstract method AbstractDatasetHandler::set_some_dictionaries "
            "should not be called."
        )

    def get_some_dictionaries(
        self, dictionary_indices: NDArray
    ) -> Iterable[MutableMapping]:
        raise NotImplementedError(
            "Abstract method AbstractDatasetHandler::get_some_dictionaries "
            "should not be called."
        )

    def set_validation_indices(self, validation_indices: NDArray) -> None:
        raise NotImplementedError(
            "Abstract method AbstractDatasetHandler::set_validation_indices "
            "should not be called."
        )

    def get_validation_indices(self) -> NDArray:
        raise NotImplementedError(
            "Abstract method AbstractDatasetHandler::get_validation_indices "
            "should not be called."
        )

    def clear_validation_indices(self) -> None:
        raise NotImplementedError(
            "Abstract method AbstractDatasetHandler::clear_validation_indices "
            "should not be called."
        )

    def set_all_label_definitions(
        self, label_definitions: Iterable[MutableMapping]
    ) -> None:
        raise NotImplementedError(
            "Abstract method AbstractDatasetHandler::set_all_label_definitions "
            "should not be called."
        )

    def get_all_label_definitions(self) -> Iterable[MutableMapping]:
        raise NotImplementedError(
            "Abstract method AbstractDatasetHandler::get_all_label_definitions "
            "should not be called."
        )

    def check_data_consistency(self) -> bool:
        raise NotImplementedError(
            "Abstract method AbstractDatasetHandler::check_data_consistency "
            "should not be called."
        )


class GenericDatasetHandler(AbstractDatasetHandler):
    def __init__(self):
        # super(GenericDatasetHandler, self).__init__()
        pass

    """
    Handle the vector of features for each stored entity.
    """

    def read_all_feature_vectors_from_h5py(
        self, filename: str, data_name: str = "features"
    ) -> None:
        """
        Read the entire database of features from a supplied h5py file.
        """
        with h5.File(filename) as ds:
            self.feature_vectors: NDArray = np.array(ds[data_name])

    def write_all_feature_vectors_to_h5py(
        self, filename: str, data_name: str = "features"
    ) -> None:
        """
        Write the entire database of features to a h5py file.
        """
        with h5.File(filename, "w") as f:
            f.create_dataset(
                data_name, self.feature_vectors.shape, data=self.feature_vectors
            )

    def set_all_feature_vectors(self, feature_vectors: NDArray) -> None:
        """
        Set the entire database of feature vectors from a supplied numpy array.
        """
        if isinstance(feature_vectors, np.ndarray) and len(feature_vectors.shape) == 2:
            self.feature_vectors = feature_vectors
        else:
            raise ValueError(
                "The argument to set_all_feature_vectors must be a 2-dimensional numpy "
                "ndarray."
            )

    def get_all_feature_vectors(self) -> NDArray:
        """
        Get the entire database of feature vectors as a numpy array.
        """
        return (
            self.feature_vectors
            if hasattr(self, "feature_vectors")
            else np.array((), dtype=np.int64)
        )

    def clear_all_feature_vectors(self) -> None:
        """
        Remove all feature vectors from the database
        """
        del self.feature_vectors

    def set_some_feature_vectors(
        self, feature_vector_indices: NDArray, feature_vectors: NDArray
    ) -> None:
        """
        This overwrites existing feature vectors.  It does not (yet) handle insert,
        delete, or append.

        N.B. as with numpy arrays in general, a single feature vector index and a list
        with one feature vector index will have different behaviors, in that the former
        drops an array dimension but the latter does not.  That is, when changing one
        feature vector use feature_vector_indices=5, feature_vectors=[3,1.2,4] OR use
        feature_vector_indices=[5], feature_vectors=[[3,1.2,4]].
        """
        self.feature_vectors[feature_vector_indices] = feature_vectors

    def get_some_feature_vectors(self, feature_vector_indices: NDArray) -> NDArray:
        """
        This fetches a subset of existing feature_vectors.

        N.B. as with numpy arrays in general, a single feature vector index and a list
        with one feature vector index will have different behaviors, in that the former
        drops an array dimension but the latter does not.  That is, use of
        feature_vector_indices=5 returns [3,1.2,4] but use of feature_vector_indices=[5]
        returns [[3,1.2,4]].
        """
        return self.feature_vectors[feature_vector_indices]

    def get_training_feature_vectors(self) -> NDArray:
        return (
            self.get_all_feature_vectors()
            if not hasattr(self, "validation_indices")
            else self.get_some_feature_vectors(
                np.fromiter(
                    set(range(self.feature_vectors.shape[0]))
                    - set(self.validation_indices),
                    dtype=np.int64,
                )
            )
        )

    def get_validation_feature_vectors(self) -> NDArray:
        return (
            np.array((), dtype=np.int64)
            if not hasattr(self, "validation_indices")
            else self.get_some_feature_vectors(self.validation_indices)
        )

    """
    Handle the label(s) for each stored entity.
    """

    def read_all_labels_from_h5py(
        self, filename: str, data_name: str = "labels"
    ) -> None:
        """
        Read the entire database of labels from a supplied h5py file.
        """
        with h5.File(filename) as ds:
            self.labels: NDArray = np.array(ds[data_name])

    def write_all_labels_to_h5py(
        self, filename: str, data_name: str = "labels"
    ) -> None:
        """
        Write the entire database of labels to a h5py file.
        """
        with h5.File(filename, "w") as f:
            f.create_dataset(data_name, self.labels.shape, data=self.labels)

    def set_all_labels(self, labels: NDArray) -> None:
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

    def get_all_labels(self) -> NDArray:
        """
        Get the entire database of labels as a numpy array.
        """
        return self.labels if hasattr(self, "labels") else np.array((), dtype=np.int64)

    def clear_all_labels(self) -> None:
        """
        Remove all labels from the database
        """
        del self.labels

    def set_some_labels(self, label_indices: NDArray, labels: NDArray) -> None:
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

    def get_some_labels(self, label_indices: NDArray) -> NDArray:
        """
        This fetches a subset of existing labels.

        N.B. as with numpy arrays in general, a single label index and a list with one
        label index will have different behaviors, in that the former drops an array
        dimension but the latter does not.  That is, use of label_indices=5 returns
        [3,1,4] but use of label_indices=[5] returns [[3,1,4]].
        """
        return self.labels[label_indices]

    def get_training_labels(self) -> NDArray:
        return (
            self.get_all_labels()
            if not hasattr(self, "validation_indices")
            else self.get_some_labels(
                np.fromiter(
                    set(range(self.labels.shape[0])) - set(self.validation_indices),
                    dtype=np.int64,
                )
            )
        )

    def get_validation_labels(self) -> NDArray:
        return (
            np.array((), dtype=np.int64)
            if not hasattr(self, "validation_indices")
            else self.get_some_labels(self.validation_indices)
        )

    def query_oracle(self, next_indices: NDArray) -> NDArray:
        """
        This method queries the oracle for labels for the supplied indices.  It returns
        all labels.  Note that in a simulation, we already have those labels and there
        is nothing to do.
        """
        return self.labels

    """
    Handle the dictionary of supplemental information for each stored entity.
    """

    def set_all_dictionaries(self, dictionaries: Iterable[MutableMapping]) -> None:
        """
        Set the entire database of dictionaries -- one per feature vector -- from a
        supplied list of Python dict objects.

        N.B. if we were handed a (read-only) tuple then we cannot change a subset of
        them.  If this functionality is needed then supply `list(dictionaries)` to this
        function.
        """
        if isinstance(dictionaries, list) and all(
            [isinstance(e, dict) for e in dictionaries]
        ):
            self.dictionaries: List[MutableMapping] = dictionaries
        else:
            raise ValueError(
                "The argument to set_all_dictionaries must be a list of Python "
                "dict objects"
            )

    def get_all_dictionaries(self) -> Iterable[MutableMapping]:
        """
        Get the entire database of dictionaries as a list (or tuple) of Python dict
        objects.
        """
        return self.dictionaries if hasattr(self, "dictionaries") else list()

    def clear_all_dictionaries(self) -> None:
        """
        Remove all dictionaries from the database
        """
        del self.dictionaries

    def set_validation_indices(self, validation_indices: NDArray) -> None:
        """
        Mark which feature vectors are reserved for validation, and should not
        participate in training.
        """
        if isinstance(validation_indices, np.ndarray):
            self.validation_indices: NDArray = validation_indices
        else:
            raise ValueError(
                "The argument to set_validation_indices must be a numpy array"
            )

    def get_validation_indices(self) -> NDArray:
        """
        Retrieve the indices of those feature vectors that are reserved for validation,
        and should not participate in training.
        """
        return (
            self.validation_indices
            if hasattr(self, "validation_indices")
            else np.array((), dtype=np.int64)
        )

    def clear_validation_indices(self) -> None:
        """
        Indicate that no feature vectors are reserved for validation.
        """
        del self.validation_indices

    def set_some_dictionaries(
        self, dictionary_indices: NDArray, dictionaries: Iterable[MutableMapping]
    ) -> None:
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

        for k, v in zip(dictionary_indices, dictionaries):
            self.dictionaries[k] = v

    def get_some_dictionaries(
        self, dictionary_indices: NDArray
    ) -> Iterable[MutableMapping]:
        """
        This fetches a subset of existing dictionaries.

        N.B. as with numpy arrays in general, a single dictionary index and a list with
        one dictionary index will have different behaviors, in that the former drops an
        array dimension but the latter does not.  That is, use of dictionary_indices=5
        returns {'a': 1, 'b': 2} but use of dictionary_indices=[5] returns [{'a': 1,
        'b': 2}].
        """
        return (
            [self.dictionaries[k] for k in dictionary_indices]
            if hasattr(self, "dictionaries")
            else list()
        )

    """
    Handle operations not specific to the feature vectors, labels, or dictionaries of
    supplemental information of each stored entity.
    """

    def set_all_label_definitions(
        self, label_definitions: Iterable[MutableMapping]
    ) -> None:
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
            self.label_definitions: List[MutableMapping] = label_definitions
        else:
            raise ValueError(
                "The argument to set_all_label_definitions must be a list of Python "
                "dict objects"
            )

    def get_all_label_definitions(self) -> Iterable[MutableMapping]:
        """
        Returns
        -------
        label_definitions: dict
            See set_all_label_definitions for the dict format.
        """
        return self.label_definitions if hasattr(self, "label_definitions") else list()

    def check_data_consistency(self) -> bool:
        # Check whether among feature vectors, labels, and dictionaries that were
        # supplied, are they for the same number of entities?
        feature_vectors_length = (
            self.feature_vectors.shape[0] if hasattr(self, "feature_vectors") else 0
        )
        labels_length = self.labels.shape[0] if hasattr(self, "labels") else 0
        dictionaries_length = (
            len(self.dictionaries) if hasattr(self, "dictionaries") else 0
        )
        # Eliminate duplicates
        all_lengths = set([feature_vectors_length, labels_length, dictionaries_length])
        lengths_test = len(all_lengths) == 1 or (
            len(all_lengths) == 2 and 0 in all_lengths
        )

        # Check whether among labels and label_definitions that were supplied, are they
        # for the same number of kinds of labels?
        labels_width = (
            (1 if len(self.labels.shape) == 1 else self.labels.shape[1])
            if hasattr(self, "labels")
            else 0
        )
        label_definitions_width = (
            len(self.label_definitions) if hasattr(self, "label_definitions") else 0
        )
        all_widths = set([labels_width, label_definitions_width])
        widths_test = len(all_widths) == 1 or (len(all_widths) == 2 and 0 in all_widths)

        # Check whether every supplied label category has a definition.
        definitions_test = (
            not widths_test  # already failed previous test
            or labels_width == 0  # nothing to compare
            or label_definitions_width == 0  # nothing to compare
            or (
                len(self.labels.shape) == 1  # 1-dimensional numpy array
                and len(set(self.labels) - set(self.label_definitions[0])) == 0
            )
            or (
                len(self.labels.shape) == 2  # 2-dimensional numpy array
                and all(
                    [
                        len(set(self.labels[:, col]) - set(self.label_definitions[col]))
                        == 0
                        for col in range(self.labels.shape[1])
                    ]
                )
            )
        )

        mesgs: List[str] = list()
        if not lengths_test:
            mesgs += [
                f"height(feature_vectors) = {feature_vectors_length}, "
                f"height(labels) = {labels_length}, and "
                f"height(dictionaries) = {dictionaries_length} do not match."
            ]
        if not widths_test:
            mesgs += [
                f"width(labels) = {labels_width} and "
                f"width(label_definitions) = {label_definitions_width} do not match."
            ]
        if not definitions_test:
            mesgs += ["Some labels have categories without definitions."]
        if len(mesgs) > 0:
            raise ValueError("\n".join(mesgs))

        return len(mesgs) == 0
