
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
        if not self.just_velocities:
            self.label_description = "[rho],[u,v],[p]"
        if tracer:
            self.label_description += ",[tracer]"

        self.pixel_mask = torch.tensor([False, False])
        if not self.just_velocities:
            self.pixel_mask = torch.tensor([False, False, False, True])
        if tracer:
            self.pixel_mask = torch.cat(
                [self.pixel_mask, torch.tensor([False])],
                dim=0,
            )

        if resolution is None:
            self.res = None
        else:
            if resolution > 128:
                raise ValueError("Resolution must be <= 128")
            self.res = resolution

        self.post_init()

    def _downsample(self, image, target_size):
        image = image.unsqueeze(0)
        image_size = image.shape[-2]
        freqs = torch.fft.fftfreq(image_size, d=1 / image_size)
        sel = torch.logical_and(freqs >= -target_size / 2, freqs <= target_size / 2 - 1)
        image_hat = torch.fft.fft2(image, norm="forward")
        image_hat = image_hat[:, :, sel, :][:, :, :, sel]
        image = torch.fft.ifft2(image_hat, norm="forward").real
        return image.squeeze(0)

    def __getitem__(self, idx):
        i, t, t1, t2 = self._idx_map(idx)
        time = t / self.constants["time"]

        inputs_v = (
            torch.from_numpy(self.reader["velocity"][i + self.start, t1, 0:2])
            .type(torch.float32)
            .reshape(2, self.resolution, self.resolution)
        )
        label_v = (
            torch.from_numpy(self.reader["velocity"][i + self.start, t2, 0:2])
            .type(torch.float32)
            .reshape(2, self.resolution, self.resolution)
        )
        if self.transpose:
            inputs_v = inputs_v.transpose(-2, -1)
            label_v = label_v.transpose(-2, -1)

        if not self.just_velocities:
            inputs = torch.cat([self.density, inputs_v, self.pressure], dim=0)
            label = torch.cat([self.density, label_v, self.pressure], dim=0)
        else:
            inputs = inputs_v
            label = label_v

        inputs = (inputs - self.constants["mean"]) / self.constants["std"]
        label = (label - self.constants["mean"]) / self.constants["std"]

        if self.tracer:
            input_tracer = (
                torch.from_numpy(self.reader["velocity"][i + self.start, t1, 2:3])
                .type(torch.float32)
                .reshape(1, self.resolution, self.resolution)
            )
            output_tracer = (
                torch.from_numpy(self.reader["velocity"][i + self.start, t2, 2:3])
                .type(torch.float32)
                .reshape(1, self.resolution, self.resolution)
            )
            if self.transpose:
                input_tracer = input_tracer.transpose(-2, -1)
                output_tracer = output_tracer.transpose(-2, -1)
            input_tracer = (
                input_tracer - self.constants["tracer_mean"]