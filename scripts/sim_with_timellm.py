import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from timellm.data_provider.data_factory import data_provider
from timellm.models import TimeLLM
from timellm.utils.tools import (
    create_checkpoint_dict,
    load_content,
    vali_pulsar,
)
from tqdm import tqdm

from tingan.datasets import partim_to_timellm_format, split_tim_and_par_files
from tingan.networks import TimeSeriesDiscriminator, trainable_parameters
from tingan.plots import (
    plot_labels,
    plot_losses,
    plot_timellm_residuals,
    plot_timing_noise,
)
from tingan.utils import set_seed

# Setting some environment variables and random seed, from Time-LLM original scripts
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

fix_seed = 2026
set_seed(fix_seed)

# Loading configuration
with Path("timellm_config.json").open() as f:
    t_args = argparse.Namespace()
    t_args.__dict__.update(json.load(f))
parser = argparse.ArgumentParser()
args = parser.parse_args(namespace=t_args)

# Checking configuration
if len(args.d_updates_per_batch) != len(args.d_updates_epochs):
    len_err_msg = (
        "d_updates_per_batch and d_updates_epochs should have the same length."
    )
    raise ValueError(len_err_msg)

if args.d_updates_epochs[0] != 0:
    d_err_msg = "d_updates_epochs should start from 0."
    raise ValueError(d_err_msg)

if args.train_epochs < args.d_updates_epochs[-1]:
    epochs_err_msg = "train_epochs should be larger than last d_updates_epochs."
    raise ValueError(epochs_err_msg)

# Necessary parameters that should not be modified
args.model = "TimeLLM"
args.data = "Pulsar"
args.llm_model = "LLAMA"
args.llm_dim = 4096
args.prompt_domain = True
args.content = load_content(args)
args.seasonal_patterns = None
args.features = "S"
args.percent = 100

# Setting up distributed training and accelerator, from Time-LLM original scripts
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

d_updates_per_batch_str, d_updates_epochs_str = "", ""
for i in range(len(args.d_updates_per_batch)):
    d_updates_per_batch_str += f"{args.d_updates_per_batch[i]}-"
    d_updates_epochs_str += f"{args.d_updates_epochs[i]}-"

# Setting record of experiments
setting = (
    f"{args.task_name}_"
    f"{args.model}_"
    f"{args.llm_model}_"
    f"{args.data}_"
    f"nr{args.nrows}_"
    f"{d_updates_per_batch_str[:-1]}_"
    f"{d_updates_epochs_str[:-1]}_"
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
    f"ei{args.enc_in}"
)

path_data = Path(args.root_path) / Path(args.data_path)
if not path_data.exists():
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

# Creating training, validation and test datasets
train_data, train_loader = data_provider(args, "train", seed=fix_seed)
vali_data, vali_loader = data_provider(args, "val", seed=fix_seed)
test_data, test_loader = data_provider(args, "test", seed=fix_seed)

# Creating generator and discriminator
model = TimeLLM.Model(args).float()
discriminator = TimeSeriesDiscriminator(
    seq_len=args.pred_len, n_channels=1 if args.features == "S" else 0
)

# Labels
real_label = torch.full((1,), 1.0, device=accelerator.device)
fake_label = torch.full((1,), 0.0, device=accelerator.device)
bce_loss = torch.nn.BCELoss()

# Creating directory where results will be saved
path = Path(args.checkpoints) / Path(setting)
if not path.exists() and accelerator.is_local_main_process:
    path.mkdir(parents=True)

with (path / Path("timellm_config.json")).open("w") as f:
    args_dict = vars(args)
    json.dump(args_dict, f, indent=4)

args.d_updates_per_batch = args.d_updates_per_batch[::-1]
args.d_updates_epochs = np.array(args.d_updates_epochs).astype(int)

fig_timellm_residuals = plot_timellm_residuals(path_data, nrows=args.nrows)
fig_timellm_residuals.savefig(path / Path("residuals.png"))

train_steps = len(train_loader)

# Optimizers and schedulter
model_optim = torch.optim.Adam(trainable_parameters(model), lr=args.learning_rate)
discr_optim = torch.optim.Adam(
    trainable_parameters(discriminator), lr=1e-4, betas=(0.5, 0.999)
)

if (path / Path("generator.pth")).exists() and (
    path / Path("discriminator.pth")
).exists():
    print("Loading checkpoint...")
    checkpoint_g = torch.load(path / Path("generator.pth"), weights_only=False)
    checkpoint_d = torch.load(path / Path("discriminator.pth"), weights_only=False)
    model.load_state_dict(checkpoint_g["model"])
    discriminator.load_state_dict(checkpoint_d["model"])

    model_optim.load_state_dict(checkpoint_g["optimizer"])
    discr_optim.load_state_dict(checkpoint_d["optimizer"])

    start_epoch = checkpoint_g["epoch"] + 1
    print("Done!")
else:
    print("Starting training from scratch.")
    start_epoch = 0

model, model_optim = accelerator.prepare(model, model_optim)
discriminator, discr_optim = accelerator.prepare(discriminator, discr_optim)

train_loss_g = []
train_loss_d = []
dlabels_for_real = []
dlabels_for_mock = []
vali_loss_d = []

d_updates_per_batch = 1

for epoch in range(start_epoch):
    set_seed(fix_seed + epoch)
    if epoch in args.d_updates_epochs:
        d_updates_per_batch = args.d_updates_per_batch.pop()
    for loader in [train_loader, vali_loader]:
        for _ in loader:
            pass

for epoch in range(start_epoch, args.train_epochs):
    set_seed(fix_seed + epoch)

    if epoch in args.d_updates_epochs:
        d_updates_per_batch = args.d_updates_per_batch.pop()

    model.train()
    discriminator.train()
    epoch_time = time.time()
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(
        enumerate(train_loader)
    ):
        # decoder input
        dec_inp = (
            torch.zeros_like(
                batch_y[:, -args.pred_len :, :].float().to(accelerator.device)
            )
            .float()
            .to(accelerator.device)
        )
        dec_inp = (
            torch.cat(
                [
                    batch_y[:, : args.label_len, :].float().to(accelerator.device),
                    dec_inp,
                ],
                dim=1,
            )
            .float()
            .to(accelerator.device)
        )

        # encoder - decoder
        outputs = model(
            batch_x.float().to(accelerator.device),
            batch_x_mark.float().to(accelerator.device),
            dec_inp,
            batch_y_mark.float().to(accelerator.device),
        )

        f_dim = -1 if args.features == "MS" else 0
        outputs = outputs[:, -args.pred_len :, f_dim:]
        batch_y_pred = (
            batch_y[:, -args.pred_len :, f_dim:].float().to(accelerator.device)
        )

        # =========================================================
        #  TRAIN DISCRIMINATOR
        # =========================================================
        for idiscr in range(d_updates_per_batch):
            set_seed(fix_seed + epoch + idiscr)
            discr_optim.zero_grad()

            # Real samples
            real_data = batch_y_pred.detach()  # (batch, pred_len, n_channels)
            d_real = discriminator(real_data)  # (batch, 1l
            # Expand labels to match batch
            labels_real = real_label.expand_as(d_real)
            loss_d_real = bce_loss(d_real, labels_real)
            dlabels_for_real.append(d_real.detach().mean().item())
            accelerator.backward(loss_d_real)

            # Fake samples
            fake_data = outputs.detach()  # detach so gradient doesn't flow to generator
            d_fake = discriminator(fake_data)
            labels_fake = fake_label.expand_as(d_fake)
            loss_d_fake = bce_loss(d_fake, labels_fake)
            dlabels_for_mock.append(d_fake.detach().mean().item())
            accelerator.backward(loss_d_fake)

            if i == 0 and idiscr == d_updates_per_batch - 1:
                fig, ax = plot_timing_noise(
                    None,
                    batch_y_pred[:, :, 0].cpu().detach().numpy(),
                    labels=d_real.cpu().detach().numpy().round(3),
                )
                fig, ax = plot_timing_noise(
                    None,
                    outputs[:, :, 0].cpu().detach().numpy(),
                    labels=d_fake.cpu().detach().numpy().round(3),
                    fig=fig,
                    ax=ax,
                )
                fig.savefig(path / Path(f"outputs_epoch{epoch}_i{i}.pdf"))

            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            discr_optim.step()

            train_loss_d.append((loss_d_real + loss_d_fake).item())

        # =========================================================
        #  TRAIN GENERATOR (Time-LLM) — MSE + Adversarial
        # =========================================================
        set_seed(fix_seed + epoch)
        model_optim.zero_grad()

        # Adversarial: we want the discriminator to think forecasts are REAL
        d_fake_for_g = discriminator(
            outputs
        )  # NO detach here — gradient flows to generator
        labels_for_g = real_label.expand_as(
            d_fake_for_g
        )  # generator wants "real" verdict
        loss_adv = bce_loss(d_fake_for_g, labels_for_g)

        loss_g = loss_adv
        train_loss_g.append(loss_g.item())

        accelerator.backward(loss_g)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        model_optim.step()

    accelerator.print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time}")
    vali_loss, vali_loss_d, vali_pred_lab, vali_true_lab = vali_pulsar(
        args, accelerator, model, discriminator, vali_data, vali_loader, bce_loss
    )
    train_loss_g.append(np.nan)
    train_loss_d.append(np.nan)
    dlabels_for_real.append(np.nan)
    dlabels_for_mock.append(np.nan)
    accelerator.print(
        f"Epoch: {epoch + 1} | Train Loss: {train_loss_g[-1]:.7f} "
        f"Train Loss D: {np.mean(train_loss_d[-d_updates_per_batch:]):.7f} "
        f"Test Loss: {vali_loss:.7f} "
        f"Test Loss D: {vali_loss_d:.7f}"
    )
    accelerator.print(
        f"\titers: {i + 1}, epoch: {epoch + 1} | "
        f"D_loss_real: {loss_d_real.item():.7f} | "
        f"D_loss_fake: {loss_d_fake.item():.7f} | "
        f"G_adv: {loss_adv.item():.7f}"
    )

    check_dict_g = create_checkpoint_dict(
        model, train_loss_g[-1], epoch, optimizer=model_optim
    )
    torch.save(check_dict_g, path / Path(f"generator_ep{epoch + 1}.pth"))
    check_dict_d = create_checkpoint_dict(
        discriminator, train_loss_d[-1], epoch, optimizer=discr_optim
    )
    torch.save(check_dict_d, path / Path(f"discriminator_ep{epoch + 1}.pth"))

accelerator.wait_for_everyone()

fig = plot_losses(train_loss_g, train_loss_d)
fig.savefig(path / Path(f"loss_ep{start_epoch + 1}-{epoch + 1}.png"))
fig = plot_labels(dlabels_for_real, dlabels_for_mock)
fig.savefig(path / Path(f"labels_ep{start_epoch + 1}-{epoch + 1}.png"))

torch.save(check_dict_g, path / Path("generator.pth"))
torch.save(check_dict_d, path / Path("discriminator.pth"))
