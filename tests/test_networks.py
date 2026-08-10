import torch

from tingan import networks


def test_discriminator_output_size() -> None:
    """Test that the output of the discriminator has the expected size."""
    timeseries_len = 128
    batch_size = 64
    discriminator = networks.TimeSeriesDiscriminator(seq_len=timeseries_len)
    time_series = torch.randn(batch_size, timeseries_len)
    output = discriminator(time_series)
    assert output.shape == torch.Size([batch_size, 1])


def test_discriminator_output_values() -> None:
    """Test that the output of the discriminator is a probability."""
    timeseries_len = 128
    batch_size = 64
    discriminator = networks.TimeSeriesDiscriminator(seq_len=timeseries_len)
    time_series = torch.randn(batch_size, timeseries_len)
    output = discriminator(time_series)
    assert (torch.max(output).item() <= 1) & (torch.min(output).item() >= 0)
