from torch import nn


class Generator(nn.Module):
    def __init__(self, nz):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(nz, 2 * nz),
            nn.ReLU(True),
            nn.Linear(2 * nz, 4 * nz),
            nn.ReLU(True),
            nn.Linear(4 * nz, 8 * nz),
            nn.ReLU(True),
            nn.Linear(8 * nz, 4 * nz),
            nn.ReLU(True),
            nn.Linear(4 * nz, 2 * nz),
            nn.ReLU(True),
            nn.Linear(2 * nz, nz),
        )

    def forward(self, random_noise):
        return self.main(random_noise)


class Discriminator(nn.Module):
    def __init__(self, nz):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(nz, 2 * nz),
            nn.ReLU(True),
            nn.Linear(2 * nz, 4 * nz),
            nn.ReLU(True),
            nn.Linear(4 * nz, 8 * nz),
            nn.ReLU(True),
            nn.Linear(8 * nz, 4 * nz),
            nn.ReLU(True),
            nn.Linear(4 * nz, 2 * nz),
            nn.ReLU(True),
            nn.Linear(2 * nz, nz),
            nn.ReLU(True),
            nn.Linear(nz, 1),
            nn.Sigmoid(),
        )

    def forward(self, noise):
        return self.main(noise)
