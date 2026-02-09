import torch


class TimingNoise(torch.utils.data.Dataset):
    def __init__(self, size):
        self.size = size
        self.std = torch.randn(self.size)

    def __getitem__(self, index):
        return self.std[index] ** 2 * torch.randn(64) + self.std[index]

    def __len__(self):
        return self.size
