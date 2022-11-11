
import torch
import h5py
import copy
from scOT.problems.base import BaseTimeDataset, BaseDataset
from scOT.problems.fluids.normalization_constants import CONSTANTS


class Airfoil(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.N_max = 10869
        self.N_val = 120
        self.N_test = 240
        self.resolution = 128

        data_path = self.data_path + "/SE-AF.nc"
        data_path = self._move_to_local_scratch(data_path)
        self.reader = h5py.File(data_path, "r")

        self.constants = {
            "mean": 0.92984116,
            "std": 0.10864315,
        }

        self.input_dim = 1