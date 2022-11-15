
import torch
import h5py
import numpy as np
import copy
from scOT.problems.base import BaseTimeDataset
from scOT.problems.fluids.normalization_constants import CONSTANTS


class IncompressibleBase(BaseTimeDataset):
    def __init__(
        self,
        N_max,
        file_path,
        *args,
        tracer=False,
        just_velocities=False,
        transpose=False,
        resolution=None,
        **kwargs
    ):
        """
        just_velocities: If True, only the velocities are used as input and output.
        transpose: If True, the input and output are transposed.
        """
        super().__init__(*args, **kwargs)
        assert self.max_num_time_steps * self.time_step_size <= 20

        self.N_max = N_max
        self.N_val = 120
        self.N_test = 240
        self.resolution = 128
        self.tracer = tracer
        self.just_velocities = just_velocities
        self.transpose = transpose

        data_path = self.data_path + file_path
        data_path = self._move_to_local_scratch(data_path)
        self.reader = h5py.File(data_path, "r")

        self.constants = copy.deepcopy(CONSTANTS)
        if just_velocities:
            self.constants["mean"] = self.constants["mean"][1:3]
            self.constants["std"] = self.constants["std"][1:3]

        self.density = torch.ones(1, self.resolution, self.resolution)
        self.pressure = torch.zeros(1, self.resolution, self.resolution)

        self.input_dim = 4 if not tracer else 5
        if just_velocities:
            self.input_dim -= 2
        self.label_description = "[u,v]"