"""Basic GAN test case."""

import random
from enum import Enum
from pathlib import Path

import numpy as np
import torch.utils.data
import yaml
from torch import nn, optim

from tingan.datasets import TimingNoise
from tingan.networks import Discriminator, Generator
from tingan.plots import plot_losses, plot_timing_noise, plot_timing_noise_properties

# Variables
with Path("config.yaml").open("r") as stream:
    config = yaml.safe_load(stream)


class TrainingLabels(Enum):
    """Defines binary labels noice classification."""

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
dataset = TimingNoise(config["train_size"])
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=config["batch_size"],
    shuffle=True,
    num_workers=config["workers"],
)

# Plot some training data
real_batch = next(iter(dataloader))
plot_timing_noise(real_batch, tin_type="Training")

# Create the generator
netg = Generator(nz=config["nz"]).to(device)

# Create the Discriminator
netd = Discriminator(nz=config["nz"]).to(device)

# Initialize the ``BCELoss`` function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
input_noise = torch.randn(config["batch_size"], config["nz"], device=device)
untrained_noise = netg(input_noise)

input_output_before_training = np.zeros((128, 64, 2))
input_output_before_training[:, :, 0] = input_noise.detach().numpy()
input_output_before_training[:, :, 1] = untrained_noise.detach().numpy()

plot_timing_noise(
    input_output_before_training, tin_type="Input and output (before training)"
)
plot_timing_noise_properties(
    (
        real_batch.detach().numpy(),
        input_noise.detach().numpy(),
        untrained_noise.detach().numpy(),
    )
)

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
    for i, data in enumerate(dataloader, 0):
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
                f"\tLoss_D: {errd.item():.4f}"
                f"\tLoss_G: {errg.item():.4f}"
                f"\tD(x): {d_x:.4f}"
                f"\tD(G(z)): {d_g_z1:.4f} / {d_g_z2:.4f}"
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

plot_losses(g_losses, d_losses)
plot_timing_noise(noise_list[-1], tin_type="Trained")
plot_timing_noise_properties((real_batch.detach().numpy(), noise_list[-1]))
