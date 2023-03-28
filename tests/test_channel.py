import pytest
import numpy as np
from video.conversion import Conversions, Converter
from video.channel import Channel


@pytest.fixture
def sample_channel():
    arr = np.random.randint(0, 256, size=(100, 50, 50), dtype=np.uint8)
    return Channel(arr, Converter(0, 0))


def test_channel_agg(sample_channel):
    result = sample_channel.agg('mean')
    assert isinstance(result, np.ndarray)
    assert result.shape == (sample_channel.channel.shape[0],)
    result = sample_channel.agg(lambda x: np.mean(x, axis=1))
    assert isinstance(result, np.ndarray)
    assert result.shape == (sample_channel.channel.shape[0],)


def test_channel_pct_change(sample_channel):
    result = sample_channel.pct_change(1)
    assert isinstance(result, np.ndarray)
    assert result.shape == (sample_channel.channel.shape[0],)


def test_channel_properties(sample_channel):
    assert sample_channel.conversion == Converter(0, 0)
    assert isinstance(sample_channel.channel, np.ndarray)
