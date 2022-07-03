
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
        per_device_eval_batch_size=batch_size,
        eval_accumulation_steps=16,
        dataloader_num_workers=workers,
    )
    time_involved = isinstance(dataset, BaseTimeDataset)

    def compute_metrics(eval_preds):
        if time_involved and output_all_steps:
            return {}
        channel_list = dataset.channel_slice_list

        def get_relative_statistics(errors):
            median_error = np.median(errors, axis=0)
            mean_error = np.mean(errors, axis=0)
            std_error = np.std(errors, axis=0)
            min_error = np.min(errors, axis=0)
            max_error = np.max(errors, axis=0)
            return {
                "median_relative_l1_error": median_error,
                "mean_relative_l1_error": mean_error,
                "std_relative_l1_error": std_error,
                "min_relative_l1_error": min_error,
                "max_relative_l1_error": max_error,
            }

        def get_statistics(errors):
            median_error = np.median(errors, axis=0)
            mean_error = np.mean(errors, axis=0)
            std_error = np.std(errors, axis=0)
            min_error = np.min(errors, axis=0)
            max_error = np.max(errors, axis=0)
            return {
                "median_l1_error": median_error,
                "mean_l1_error": mean_error,
                "std_l1_error": std_error,
                "min_l1_error": min_error,
                "max_l1_error": max_error,
            }

        relative_errors = [
            relative_lp_error(
                eval_preds.predictions[:, channel_list[i] : channel_list[i + 1]],
                eval_preds.label_ids[:, channel_list[i] : channel_list[i + 1]],
                p=1,
                return_percent=True,
            )
            for i in range(len(channel_list) - 1)
        ]

        errors = [
            lp_error(
                eval_preds.predictions[:, channel_list[i] : channel_list[i + 1]],
                eval_preds.label_ids[:, channel_list[i] : channel_list[i + 1]],
                p=1,
            )
            for i in range(len(channel_list) - 1)
        ]

        relative_error_statistics = [
            get_relative_statistics(relative_errors[i])
            for i in range(len(channel_list) - 1)
        ]

        error_statistics = [
            get_statistics(errors[i]) for i in range(len(channel_list) - 1)
        ]

        if dataset.output_dim == 1:
            relative_error_statistics = relative_error_statistics[0]
            error_statistics = error_statistics[0]
            if full_data:
                relative_error_statistics["relative_full_data"] = relative_errors[
                    0
                ].tolist()
                error_statistics["full_data"] = errors[0].tolist()
            return {**relative_error_statistics, **error_statistics}
        else:
            mean_over_relative_means = np.mean(
                np.array(
                    [
                        stats["mean_relative_l1_error"]
                        for stats in relative_error_statistics
                    ]
                ),
                axis=0,
            )
            mean_over_relative_medians = np.mean(
                np.array(
                    [
                        stats["median_relative_l1_error"]
                        for stats in relative_error_statistics
                    ]
                ),
                axis=0,
            )
            mean_over_means = np.mean(
                np.array([stats["mean_l1_error"] for stats in error_statistics]), axis=0
            )
            mean_over_medians = np.mean(
                np.array([stats["median_l1_error"] for stats in error_statistics]),
                axis=0,
            )

            error_statistics_ = {
                "mean_relative_l1_error": mean_over_relative_means,
                "mean_over_median_relative_l1_error": mean_over_relative_medians,
                "mean_l1_error": mean_over_means,
                "mean_over_median_l1_error": mean_over_medians,
            }
            #!! The above is different from train and finetune (here mean_relative_l1_error is mean over medians instead of mean over means)
            for i, stats in enumerate(relative_error_statistics):
                for key, value in stats.items():
                    error_statistics_[
                        dataset.printable_channel_description[i] + "/" + key
                    ] = value
                    if full_data:
                        error_statistics_[
                            dataset.printable_channel_description[i]
                            + "/"
                            + "relative_full_data"
                        ] = relative_errors[i].tolist()
            for i, stats in enumerate(error_statistics):
                for key, value in stats.items():
                    error_statistics_[
                        dataset.printable_channel_description[i] + "/" + key
                    ] = value
                    if full_data:
                        error_statistics_[
                            dataset.printable_channel_description[i] + "/" + "full_data"
                        ] = errors[i].tolist()
            return error_statistics_

    trainer = Trainer(
        model=model,