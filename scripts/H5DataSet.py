import h5py as h5
import numpy as np

"""
H5DataSet class

Reads an h5py file to load a Python dictionary
"""


class H5DataSet:
    def __init__(self, filename=None):
        self.filename = None
        self.dataset = None
        self.load_dataset(filename)

    def load_dataset(self, filename=None):
        if self.filename != filename:
            self.filename = filename
            if self.filename is not None:
                with h5.File(filename) as file_handle:
                    self.dataset = {
                        key: np.array(value) if len(value.shape) > 0 else value[()]
                        for (key, value) in file_handle.items()
                    }
            else:
                self.dataset = None
        return self.dataset

    def get_filename(self):
        return self.filename

    def get_dataset(self):
        return self.dataset
