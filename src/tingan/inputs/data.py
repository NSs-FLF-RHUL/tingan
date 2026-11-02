"""tingan inputs data."""

from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
from timellm.data_provider.data_factory import data_provider
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from tingan.datasets import (
    fake_dataset_timellm_format,
    partim_to_timellm_format,
    split,
    split_tim_and_par_files,
    toas_to_timellm_format,
)
from tingan.gp_rednoise import gp_dataset_for_timellm


def load_timellm_dataset(
    args: Namespace,
) -> tuple[Dataset, DataLoader, Dataset, DataLoader]:
    """
    Load timellm datasets.

    Training and validation datasets are loaded according to run arguments. If needed,
    datasets are created, provided that relevant files exist. The following options are
    available:
      - a regular dataset, created from an irregular dataset
      - a red noise dataset, from real JBO data
      - a red noise dataset, from fake data
      - a residual datasets, from fake data
      - a residual datasets, from real data fitted with TEMPO2
      - a residual datasets, from real JBO data

    :param args: run arguments
    """
    td, vd = None, None
    path_data = compute_path_data(args)

    if args.data_path == "" or not path_data.exists():
        if "JBO" in args.root_path:
            args, td, vd = gp_dataset_for_timellm(args, save_npy=False)
        if "_gp.csv" in args.data_path:
            create_fake_gp_dataset(args)
        elif len(args.data_path.split("_")) == 4:
            create_fake_math_dataset(args)
        elif path_data.with_suffix(".reswerr").exists():
            create_dataset_from_tempo2_residuals(args)
        elif (
            path_data.with_suffix(".tim").exists()
            and path_data.with_suffix(".par").exists()
        ):
            create_dataset_from_tim_and_par_files(args)

    return create_timellm_datasets(args, td, vd)


def create_timellm_datasets(
    args: Namespace,
    train_data: Dataset | list[Dataset] | None = None,
    vali_data: Dataset | list[Dataset] | None = None,
) -> tuple[Dataset, DataLoader, Dataset, DataLoader]:
    """
    Create training and validation datasets in timellm format.

    If lists of training/validation datasets are provided, they are first concatenated.
    Dataloaders are then created, and data are loaded according to the run arguments
    if no dataset is provided. Lastly, we check that the dataset length does not
    exceed the requirements, and subsample the datasets if needed.

    :param args: run arguments
    :param train_data: (list of) torch dataset(s) for training, optional
    :param vali_data: (list of) torch dataset(s) for validation, optional
    """
    if type(train_data) is list:
        train_data = ConcatDataset(train_data)
    if type(vali_data) is list:
        vali_data = ConcatDataset(vali_data)

    train_data, train_loader = data_provider(
        args,
        "train",
        seed=args.seed if args.use_seed else None,
        data_set=train_data,
    )
    vali_data, vali_loader = data_provider(
        args,
        "val",
        seed=args.seed if args.use_seed else None,
        data_set=vali_data,
    )

    ntot = len(train_data) + len(vali_data)
    if ntot > args.nrows:
        nt = int(len(train_data) / ntot * args.nrows)
        nv = int(len(vali_data) / ntot * args.nrows)
        train_data = Subset(train_data, range(nt))
        vali_data = Subset(vali_data, range(nv))
        return create_timellm_datasets(args, train_data, vali_data)

    return train_data, train_loader, vali_data, vali_loader


def create_fake_gp_dataset(args: Namespace) -> None:
    """Create fake red noise dataset."""
    gp = np.load(Path(args.root_path) / "gp4096.npy")[124]
    fake_dataset_timellm_format(
        time_start="1990-01-02T10:00:00",
        nt=len(gp) // 2,
        delta=int(24 * 3600 * (gp[1] - gp[0])),
        dist=gp[len(gp) // 2 :],
        path=args.root_path,
    )


def create_fake_math_dataset(args: Namespace) -> None:
    """Create fake residual dataset."""
    fake_dataset_timellm_format(
        time_start=args.data_path.split("_")[0],
        nt=int(float(args.data_path.split("_")[1])),
        delta=int(float(args.data_path.split("_")[2])),
        dist=args.data_path.split("_")[3][:-4],
        path=args.root_path,
    )


def create_dataset_from_tempo2_residuals(args: Namespace) -> None:
    """Create dataset from TEMPO2 residuals."""
    path_data = compute_path_data(args)
    frame = np.loadtxt(
        path_data.with_suffix(".reswerr"),
        delimiter=" ",
        skiprows=21,
    )
    frame = pd.DataFrame(
        np.array(
            [
                toas_to_timellm_format(frame[:, 0]),
                frame[:, 4],
                1e-6 * frame[:, 2],
            ]
        ).T
    )
    frame.to_csv(path_data, header=["date", "resid_s", "err_s"], index=False)


def create_dataset_from_tim_and_par_files(args: Namespace) -> None:
    """Create dataset from real JBO data."""
    path_data = compute_path_data(args)
    n = split_tim_and_par_files(
        path_data.with_suffix(".tim"), path_data.with_suffix(".par")
    )
    [*_, prefix, _] = args.data_path.split(".")
    dfs = [
        partim_to_timellm_format(
            Path(args.root_path) / Path(f"{prefix}_{i}").with_suffix(".par"),
            Path(args.root_path) / Path(f"{prefix}_{i}").with_suffix(".tim"),
        )
        for i in range(n)
    ]
    frame = pd.concat(dfs, axis=0, ignore_index=True)
    frame.to_csv(path_data, header=["date", "resid_s", "err_s"], index=False)


def create_regular_dataset_from_csv(args: Namespace) -> tuple[Namespace, list, list]:
    """Create regular dataset from irregular dataset stored in csv file."""
    path_data = compute_path_data(args)
    split_pars = args.data_path[:-4].split("_")
    ichunk = split(
        str(path_data)[:-33] + ".csv",
        tsamp=float(split_pars[-3].split("tsamp")[1]),
        tol=float(split_pars[-2].split("tol")[1]),
        min_len=int(float(split_pars[-1].split("atleast")[1])),
    )
    td, vd = [], []
    local_root = args.data_path[:-4]
    for iii in range(ichunk):
        args.data_path = f"{local_root}_chunk{iii}.csv"
        train_data, _, vali_data, _ = create_timellm_datasets(args)
        td.append(train_data)
        vd.append(vali_data)

    return args, td, vd


def compute_path_data(args: Namespace) -> Path:
    """Compute absolute data path from run arguments."""
    return Path(args.root_path) / Path(args.data_path)
