
"""
Use this script for inference/testing a scOT model.
The script can be used in different modes:
- save_samples: Save samples from a model.
- save_samples_sweep: Save samples from a sweep.
- eval: Evaluate a model on the test set.
- eval_sweep: Evaluate a sweep on the test set.
- eval_accumulation_error: Evaluate the accumulation error of a model.
- eval_resolutions: Evaluate a model on different resolutions.

See the --help page for more information.
"""

import argparse
import torch
import numpy as np
import random
import psutil
import os
import pandas as pd
import wandb
from transformers.trainer_utils import EvalPrediction
from scOT.model import ScOT
from scOT.trainer import TrainingArguments, Trainer
from scOT.problems.base import get_dataset, BaseTimeDataset
from scOT.metrics import relative_lp_error, lp_error


SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


def get_trainer(
    model_path,
    batch_size,
    dataset,
    full_data=False,
    output_all_steps=False,
    workers=-1,
):
    """
    Get a trainer for the model (actually just using the interface for inference).

    Args:
        model_path: str
            Path to the model.
        batch_size: int
            Batch size for evaluation.
        dataset: BaseTimeDataset
            Test set.
        full_data: bool
            Whether to save the full data distribution.
        output_all_steps: bool
            Whether to output all preliminary steps in autoregressive rollout.
        workers: int
            Number of workers for evaluation. If -1 will use all available cores.
    """
    num_cpu_cores = len(psutil.Process().cpu_affinity())
    if workers == -1:
        workers = num_cpu_cores
    if workers > num_cpu_cores:
        workers = num_cpu_cores
    assert workers > 0

    model = ScOT.from_pretrained(model_path)
    args = TrainingArguments(
        output_dir=".",