
"""Utility functions."""


def read_cli(parser):
    """Reads command line arguments."""

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file or JSON string",
    )
    parser.add_argument(
        "--json_config",
        action="store_true",
        help="Whether the config is a JSON string",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        required=False,
        default=None,
        help="Name of the run in wandb",
    )
    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default="scOT",
        help="Name of the wandb project",
    )
    parser.add_argument(
        "--max_num_train_time_steps",
        type=int,
        default=None,
        help="Maximum number of time steps to use for training and validation.",
    )
    parser.add_argument(
        "--train_time_step_size",
        type=int,
        default=None,
        help="Time step size to use for training and validation.",
    )
    parser.add_argument(
        "--train_small_time_transition",
        action="store_true",
        help="Whether to train only for next step prediction.",
    )