
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
        self.label_description = "[rho]"

        self.post_init()

    def __getitem__(self, idx):
        i = idx
        inputs = (
            torch.from_numpy(self.reader["solution"][i + self.start, 0])
            .type(torch.float32)
            .reshape(1, self.resolution, self.resolution)
        )
        labels = (
            torch.from_numpy(self.reader["solution"][i + self.start, 1])
            .type(torch.float32)
            .reshape(1, self.resolution, self.resolution)
        )

        labels = (labels - self.constants["mean"]) / self.constants["std"]

        pixel_mask = inputs == 1
        labels[pixel_mask] = 1

        return {
            "pixel_values": inputs,
            "labels": labels,
            "pixel_mask": pixel_mask,
        }


class RichtmyerMeshkov(BaseTimeDataset):
    def __init__(self, *args, tracer=False, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.max_num_time_steps * self.time_step_size <= 20

        self.N_max = 1260
        self.N_val = 100
        self.N_test = 130
        self.resolution = 128

        data_path = self.data_path + "/CE-RM.nc"
        data_path = self._move_to_local_scratch(data_path)
        self.reader = h5py.File(data_path, "r")

        self.constants = {
            "mean": torch.tensor([1.1964245, -7.164812e-06, 2.8968952e-06, 1.5648036])
            .unsqueeze(1)
            .unsqueeze(1),
            "std": torch.tensor([0.5543239, 0.24304213, 0.2430597, 0.89639103])
            .unsqueeze(1)
            .unsqueeze(1),
            "time": 20.0,
        }

        self.input_dim = 4
        self.label_description = "[rho],[u,v],[p]"

        self.pixel_mask = torch.tensor([False, False, False, False])
