
"""
This file contains the dataset selector get_dataset, as well as the base 
classes for all datasets.
"""

from torch.utils.data import Dataset, ConcatDataset
from typing import Optional, List, Dict
from abc import ABC
import re
import os
import shutil
from accelerate.utils import broadcast_object_list


def get_dataset(dataset, **kwargs):
    """
    Get a dataset by name.
    If you enter a list of str, will return a ConcatDataset of the datasets.

    Available choices are:
    - fluids.incompressible.BrownianBridge(.tracer)
    - fluids.incompressible.Gaussians(.tracer)
    - fluids.incompressible.ShearLayer
    - fluids.incompressible.Sines(.tracer)
    - fluids.incompressible.PiecewiseConstants(.tracer)
    - fluids.incompressible.VortexSheet(.tracer)
    - fluids.incompressible.forcing.KolmogorovFlow
    - fluids.compressible.gravity.RayleighTaylor(.tracer)
    - fluids.compressible.RiemannKelvinHelmholtz
    - fluids.compressible.RiemannCurved
    - fluids.compressible.Riemann
    - fluids.compressible.KelvinHelmholtz
    - fluids.compressible.Gaussians
    - fluids.compressible.RichtmyerMeshkov(.tracer)
    - fluids.compressible.steady.Airfoil(.time)
    - elliptic.poisson.Gaussians(.time)
    - elliptic.Helmholtz(.time)
    - wave.Layer
    - wave.Gaussians
    - reaction_diffusion.AllenCahn

    Adding .out at the end of the str, returns a dataset with more time steps.
    **kwargs overwrite the default settings.
    .time is a time-wrapped time-independent dataset.
    """
    if isinstance(dataset, list):
        return ConcatDataset([get_dataset(d, **kwargs) for d in dataset])
    if "fluids" in dataset:
        if "fluids.incompressible" in dataset:
            if "BrownianBridge" in dataset:
                from .fluids.incompressible import BrownianBridge as dset
            elif "Gaussians" in dataset:
                from .fluids.incompressible import Gaussians as dset
            elif "ShearLayer" in dataset:
                from .fluids.incompressible import ShearLayer as dset
            elif "Sines" in dataset:
                from .fluids.incompressible import Sines as dset
            elif "PiecewiseConstants" in dataset:
                from .fluids.incompressible import PiecewiseConstants as dset
            elif "VortexSheet" in dataset:
                from .fluids.incompressible import VortexSheet as dset
            elif "forcing" in dataset:
                if "KolmogorovFlow" in dataset:
                    from .fluids.incompressible import KolmogorovFlow as dset
                else:
                    raise ValueError(f"Unknown dataset {dataset}")
            else:
                raise ValueError(f"Unknown dataset {dataset}")
        elif "fluids.compressible" in dataset:
            if "gravity" in dataset:
                if "RayleighTaylor" in dataset:
                    from .fluids.compressible import RayleighTaylor as dset

                    if "out" in dataset:
                        default_time_settings = {
                            "max_num_time_steps": 10,
                            "time_step_size": 1,
                        }
                    else:
                        default_time_settings = {
                            "max_num_time_steps": 7,
                            "time_step_size": 1,
                        }
                    kwargs = {**default_time_settings, **kwargs}
                elif "Blast" in dataset:
                    from .fluids.compressible import Blast as dset
            elif "RiemannKelvinHelmholtz" in dataset:
                from .fluids.compressible import RiemannKelvinHelmholtz as dset
            elif "RiemannCurved" in dataset:
                from .fluids.compressible import RiemannCurved as dset
            elif "Riemann" in dataset:
                from .fluids.compressible import Riemann as dset
            elif "KelvinHelmholtz" in dataset:
                from .fluids.compressible import KelvinHelmholtz as dset
            elif "Gaussians" in dataset:
                from .fluids.compressible import Gaussians as dset
            elif "RichtmyerMeshkov" in dataset:
                from .fluids.compressible import RichtmyerMeshkov as dset
            elif "steady" in dataset:
                if "steady.Airfoil" in dataset:
                    from .fluids.compressible import Airfoil as dset

                    if "out" in dataset:
                        raise ValueError(f"Unknown dataset {dataset}")
                else:
                    raise ValueError(f"Unknown dataset {dataset}")
            else:
                raise ValueError(f"Unknown dataset {dataset}")
        else:
            raise ValueError(f"Unknown dataset {dataset}")
        if "out" in dataset:
            default_time_settings = {"max_num_time_steps": 10, "time_step_size": 2}
        else:
            default_time_settings = {"max_num_time_steps": 7, "time_step_size": 2}
        if "tracer" in dataset:
            tracer = True
        else:
            tracer = False
        if not "steady" in dataset:
            kwargs = {"tracer": tracer, **default_time_settings, **kwargs}
    elif "elliptic" in dataset:
        if ".out" in dataset:
            raise NotImplementedError(f"Unknown dataset {dataset}")
        if "elliptic.poisson" in dataset:
            if "Gaussians" in dataset:
                from .elliptic.poisson import Gaussians as dset
            else:
                raise ValueError(f"Unknown dataset {dataset}")
        elif "elliptic.Helmholtz" in dataset:
            from .elliptic.helmholtz import Helmholtz as dset
        else:
            raise ValueError(f"Unknown dataset {dataset}")
    elif "wave" in dataset:
        if "wave.Layer" in dataset:
            if "out" in dataset:
                default_time_settings = {"max_num_time_steps": 10, "time_step_size": 2}
            else:
                default_time_settings = {"max_num_time_steps": 7, "time_step_size": 2}
            kwargs = {**default_time_settings, **kwargs}
            from .wave.acoustic import Layer as dset
        elif "wave.Gaussians" in dataset:
            if "out" in dataset:
                raise ValueError(f"Unknown dataset {dataset}")
            else:
                default_time_settings = {"max_num_time_steps": 7, "time_step_size": 2}
            kwargs = {**default_time_settings, **kwargs}
            from .wave.acoustic import Gaussians as dset
        else:
            raise ValueError(f"Unknown dataset {dataset}")
    elif "reaction_diffusion" in dataset:
        if "reaction_diffusion.AllenCahn" in dataset:
            if "out" in dataset:
                default_time_settings = {"max_num_time_steps": 9, "time_step_size": 2}
            else:
                default_time_settings = {"max_num_time_steps": 7, "time_step_size": 2}
            kwargs = {**default_time_settings, **kwargs}
            from .reaction_diffusion.allen_cahn import AllenCahn as dset
    else:
        raise ValueError(f"Unknown dataset {dataset}")

    return dset(**kwargs) if ".time" not in dataset else TimeWrapper(dset(**kwargs))


class BaseDataset(Dataset, ABC):
    """A base class for all datasets. Can be directly derived from if you have a steady/non-time dependent problem."""

    def __init__(
        self,
        which: Optional[str] = None,
        num_trajectories: Optional[int] = None,
        data_path: Optional[str] = "./data",
        move_to_local_scratch: Optional[str] = None,
    ) -> None:
        """