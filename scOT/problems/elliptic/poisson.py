import torch
import os
import h5py
from scOT.problems.base import BaseDataset

CONSTANTS = {
    "mean_source": 0.014822142414492256,
    "std_source": 4.755138816607612,
    "mean_solution": 0.00056034584349