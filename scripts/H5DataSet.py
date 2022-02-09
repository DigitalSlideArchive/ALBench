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

    def getSlideIdx(self, slide):
        idx = np.argwhere(self.dataset["slides"] == slide)[0, 0]
        return idx

    def getDataIdx(self, index):
        return self.dataset["dataIdx[index][0]"]

    def getObjNum(self, index):
        if self.dataset["n_slides"] > index + 1:
            num = self.dataset["dataIdx"][index + 1, 0]
        else:
            num = self.dataset["n_objects"]
        return num - self.dataset["dataIdx"][index, 0]

    def getFeatureSet(self, index, num):
        return self.dataset["features"][index : index + num]

    def getWSI_Mean(self, index):
        return self.dataset["wsi_mean"][index][:]

    def getWSI_Std(self, index):
        return self.dataset["wsi_stddev"][index][:]

    def getXcentroidSet(self, index, num):
        return self.dataset["x_centroid"][index : index + num]

    def getYcentroidSet(self, index, num):
        return self.dataset["y_centroid"][index : index + num]

    def getSlideIdxSet(self, index, num):
        return self.dataset["slideIdx"][index : index + num]
