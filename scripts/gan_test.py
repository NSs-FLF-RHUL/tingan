"""Basic GAN test case."""

import argparse
import random
from enum import Enum
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
import yaml
from torch import nn, optim

from tingan.datasets import RealTimingNoise, TimingNoise
from tingan.networks import Discriminator, Generator
from tingan.plots import plot_losses, plot_timing_noise, plot_timing_noise_properties

parser = argparse.ArgumentParser()
parser.add_argument(
    "-c",
    "--config",
    default="configs/basic_test_config.yaml",
    help="Path to run configuration file.",
)
parser.add_argument("-i", "--ic", default=False, type=bool)
args = parser.parse_args()

# Variables
with Path(args.config).open("r") as stream:
    config = yaml.safe_load(stream)


class TrainingLabels(Enum):
    """Defines binary labels noise classification."""

    FAKE = 0
    REAL = 1


# For reproducible results
if config["manualSeed"] is not None:
    random.seed(int(config["manualSeed"]))
    torch.manual_seed(int(config["manualSeed"]))
    torch.use_deterministic_algorithms(mode=True)

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create the dataloader
if args.config == "configs/basic_test_config.yaml":
    dataset = TimingNoise(config["train_size"])
    input_noise = torch.randn(config["batch_size"], config["nz"], device=device)
else:
    dataset = RealTimingNoise(use_inverse_cumsum=args.ic)
    input_noise = torch.randn(config["batch_size"], 2, config["nz"], device=device)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=config["batch_size"],
    shuffle=False,
    num_workers=config["workers"],
)

# Plot some training data
real_batch = next(iter(dataloader))
train_noise_plot = plot_timing_noise(dataset, real_batch, tin_type="Training")

# Create the generator
netg = Generator(nz=config["nz"]).to(device)

# Create the Discriminator
netd = Discriminator(nz=config["nz"]).to(device)

# Initialize the ``BCELoss`` function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
untrained_noise = netg(input_noise)

noise_plot, ax = plot_timing_noise(
    dataset,
    input_noise.detach().cpu().numpy(),
    tin_type="Input and output (before training)",
)
noise_plot, ax = plot_timing_noise(
    dataset,
    untrained_noise.detach().cpu().numpy(),
    tin_type="Input and output (before training)",
    fig=noise_plot,
    ax=ax,
)

if args.config == "configs/basic_test_config.yaml":
    input_output_before_training = np.zeros((config["batch_size"], config["nz"], 2))
    input_output_before_training[:, :, 0] = input_noise.detach().cpu().numpy()
    input_output_before_training[:, :, 1] = untrained_noise.detach().cpu().numpy()
    noise_prop_plot = plot_timing_noise_properties(
        (
            real_batch.detach().cpu().numpy(),
            input_noise.detach().cpu().numpy(),
            untrained_noise.detach().cpu().numpy(),
        )
    )
    noise_prop_plot.show()
else:
    input_output_before_training = np.zeros((config["batch_size"], config["nz"], 4))
    input_output_before_training[:, :, :2] = (
        input_noise.detach()
        .cpu()
        .numpy()
        .reshape((config["batch_size"], config["nz"], 2))
    )  # [:,:2,:]
    input_output_before_training[:, :, 2:] = (
        untrained_noise.detach()
        .cpu()
        .numpy()
        .reshape((config["batch_size"], config["nz"], 2))
    )  # [:,0,:]


for fig in [train_noise_plot[0], noise_plot]:
    fig.show()

# Setup Adam optimizers for both G and D
optimizerd = optim.Adam(
    netd.parameters(), lr=config["lr"], betas=(config["beta1"], 0.999)
)
optimizerg = optim.Adam(
    netg.parameters(), lr=config["lr"], betas=(config["beta1"], 0.999)
)

# Lists to keep track of progress
noise_list = []
g_losses = []
d_losses = []
iters = 0

# For each epoch
for epoch in range(config["num_epochs"]):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netd.zero_grad()
        # Format batch
        real_cpu = data.to(device)
        b_size = real_cpu.size(0)
        label = torch.full(
            (b_size,), TrainingLabels.REAL.value, dtype=torch.float, device=device
        )
        # Forward pass real batch through D
        output = netd(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errd_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errd_real.backward()
        d_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, config["nz"], device=device)
        # Generate fake data with G
        fake = netg(noise)
        label.fill_(TrainingLabels.FAKE.value)
        # Classify all fake batch with D
        output = netd(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errd_fake = criterion(output, label)
        # Calculate the gradients for this batch,
        # accumulated (summed) with previous gradients
        errd_fake.backward()
        d_g_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errd = errd_real + errd_fake
        # Update D
        optimizerd.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netg.zero_grad()
        label.fill_(
            TrainingLabels.REAL.value
        )  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass
        # of all-fake batch through D
        output = netd(fake).view(-1)
        # Calculate G's loss based on this output
        errg = criterion(output, label)
        # Calculate gradients for G
        errg.backward()
        d_g_z2 = output.mean().item()
        # Update G
        optimizerg.step()

        # Output training stats
        if i % 50 == 0:
            print(
                f"[{epoch:d}/{config['num_epochs']:d}]][{i:d}/{len(dataloader):d}]]"
                f"\tLoss_D: {errd.item():.4f}"  # discriminator's loss
                f"\tLoss_G: {errg.item():.4f}"  # generator's loss
                f"\tD(x): {d_x:.4f}"  # mean output value of discriminator on real data,
                # should be close to 0.5
                f"\tD(G(z)): {d_g_z1:.4f} / {d_g_z2:.4f}"  # same but on fake data,
                # before and after network update
            )

        # Save Losses for plotting later
        g_losses.append(errg.item())
        d_losses.append(errd.item())

        # Check how the generator is doing by saving G's output on input_noise
        if (iters % 500 == 0) or (
            (epoch == config["num_epochs"] - 1) and (i == len(dataloader) - 1)
        ):
            with torch.no_grad():
                fake = netg(input_noise).detach().cpu().numpy()
            noise_list.append(fake)

        iters += 1

loss_plot = plot_losses(g_losses, d_losses)
noise_plot = plot_timing_noise(dataset, noise_list[-1], tin_type="Trained")
if args.config == "configs/basic_test_config.yaml":
    noise_prop_plot = plot_timing_noise_properties(
        (real_batch.detach().cpu().numpy(), noise_list[-1])
    )

for fig in [loss_plot, noise_plot[0]]:
    fig.show()
plt.show()
