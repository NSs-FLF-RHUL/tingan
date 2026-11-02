"""tingan inputs arguments."""

import argparse
import json
from pathlib import Path

from timellm.utils.tools import load_content
from torch import initial_seed

from tingan.utils import updates_str


def load_config(script: str | None = None) -> argparse.Namespace:
    """
    Load run configuration.

    The name of the config (JSON) file should be provided as CL argument
    with flag -i. All other arguments are read from the file.

    :param script: Name of the script that the configuration is to be loaded.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_file",
        required=False,
        default="configs/timellm_config.json",
        type=str,
        help="Input file",
    )
    args = parser.parse_args()
    with Path(args.input_file).open() as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))

    if script == "sim_with_timellm.py":
        set_fixed_sim_with_timellm_args(t_args)

    return parser.parse_args(namespace=t_args)


def set_fixed_sim_with_timellm_args(args: argparse.Namespace) -> None:
    """
    Set fixed arguments for the sim_with_timellm script.

    These fixed arguments are not arguments, properly speaking, for sim_with_timellm
    but timellm needs them to be present.

    :param args: run arguments.
    """
    args.model = "TimeLLM"
    args.data = "Pulsar"
    args.llm_model = "LLAMA"
    args.llm_dim = 4096
    args.prompt_domain = True
    if args.content is None:
        args.content = load_content(args)
    args.seasonal_patterns = None
    args.features = "S"
    args.percent = 100
    if args.seed is None:
        args.seed = initial_seed() % 2**32


def setting_str_from_args(args: argparse.Namespace) -> str:
    """
    Create a setting string from run arguments.

    :param args: run arguments.
    """
    d_updates_per_batch_str, d_updates_epochs_str = updates_str(
        args.d_updates_per_batch, args.d_updates_epochs
    )
    g_updates_per_batch_str, _ = updates_str(
        args.g_updates_per_batch, args.d_updates_epochs
    )

    return (
        f"{args.task_name}_"
        f"{args.model}_"
        f"{args.llm_model}_"
        f"{args.data}_"
        f"nr{args.skiprows}"
        f"-{args.nrows}_"
        f"{d_updates_per_batch_str}_"
        f"{g_updates_per_batch_str}_"
        f"{d_updates_epochs_str}_"
        f"bs{args.batch_size}_"
        f"sl{args.seq_len}_"
        f"ll{args.label_len}_"
        f"pl{args.pred_len}_"
        f"ptl{args.patch_len}_"
        f"ll{args.llm_layers}_"
        f"do{args.dropout}_"
        f"sd{args.stride}_"
        f"lr{args.learning_rate}_"
        f"{args.lradj}_"
        f"ps{args.pct_start}_"
        f"dm{args.d_model}_"
        f"nh{args.n_heads}_"
        f"df{args.d_ff}_"
        f"eb{args.embed}_"
        f"ei{args.enc_in}_"
        f"se{args.seed}"
    )
