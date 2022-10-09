import torch
import os
import h5py
from scOT.problems.base import BaseDataset

CONSTANTS = {
    "mean_source": 0.014822142414492256,
    "std_source": 4.755138816607612,
    "mean_solution": 0.0005603458434937093,
    "std_solution": 0.02401226126952699,
}


class Gaussians(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.N_max = 20000
        self.N_