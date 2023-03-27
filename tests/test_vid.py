import pytest
import numpy as np
from video.conversion import Conversions
from video.generator import vid_frame
from video.reader import VideoReader
from video.vid import Channel, Frame, Video, AGG_FUNCS


@pytest.fixture
def sample_hls_video():
    arr = np.random.randint(0, 256, size=(100, 50, 50, 3), dtype=np.uint8)
    return Video(arr, fps=30, converter=Conversions.HLS)


@pytest.fixture
def sample_hsv_video():
    arr = np.random.randint(0, 256, size=(100, 50, 50, 3), dtype=np.uint8)
    return Video(arr, fps=30, converter=Conversions.HSV)


@pytest.fixture
def sample_rgb_video():
    arr = np.random.randint(0, 256, size=(100, 50, 50, 3), dtype=np.uint8)
    return Video(arr, fps=30, converter=Conversions.RGB)


@pytest.fixture
def sample_bgr_video():
    arr = np.random.randint(0, 256, size=(100, 50, 50, 3), dtype=np.uint8)
    return Video(arr, fps=30, converter=Conversions.BGR)


@pytest.fixture
def sample_frame():
    frame_arr = np.random.randint(0, 256, size=(50, 50, 3), dtype=np.uint8)
    return Frame(frame_arr, fps=30, conversion=Conversions.HLS.value,
                 frame_no=10)


def test_channel_agg(sample_channel):
    result = sample_channel.agg('mean')
    assert isinstance(result, np.ndarray)
    assert result.shape == (sample_channel.channel.shape[0],)
    result = sample_channel.agg(AGG_FUNCS['mean'])
    assert isinstance(result, np.ndarray)
    assert result.shape == (sample_channel.channel.shape[0],)


def test_channel_pct_change(sample_channel):
    result = sample_channel.pct_change(1)
    assert isinstance(result, np.ndarray)
    assert result.shape == (sample_channel.channel.shape[0],)


def test_video_properties(sample_hls_video, sample_hsv_video,
                          sample_bgr_video, sample_rgb_video):
    assert isinstance(sample_hls_video.hue, Channel)
    assert isinstance(sample_hls_video.lightness, Channel)
    assert isinstance(sample_hls_video.saturation, Channel)
    assert sample_hls_video.width == 50
    assert sample_hls_video.height == 50
    assert sample_hls_video.frame_count == 100
    assert sample_hls_video.fps == 30
    assert isinstance(sample_hsv_video.hue, Channel)
    assert isinstance(sample_hsv_video.saturation, Channel)
    assert isinstance(sample_hsv_video.value, Channel)
    assert isinstance(sample_bgr_video.blue, Channel)
    assert isinstance(sample_bgr_video.green, Channel)
    assert isinstance(sample_bgr_video.red, Channel)
    assert isinstance(sample_rgb_video.red, Channel)
    assert isinstance(sample_rgb_video.green, Channel)
    assert isinstance(sample_rgb_video.blue, Channel)
    with pytest.raises(Exception):
        sample_hls_video.red
        sample_hsv_video.green
        sample_bgr_video.hue
        sample_rgb_video.saturation


def test_video_from_file():
    video_path = "test_vid.mp4"
    video = Video.from_file("tests/test_vid.mp4")
    assert isinstance(video, Video)
    assert video.frame_count > 0


def test_video_getitem(sample_hls_video):
    print(type(sample_hls_video))
    result = sample_hls_video[0]
    assert isinstance(result, Frame)
    result = sample_hls_video[:10]
    assert isinstance(result, Video)
    assert result.frame_count == 10


def test_video_setitem(sample_hls_video):
    sample_hls_video[0] = 128
    assert np.all(sample_hls_video[0]._vid == 128)
    with pytest.raises(Exception):
        sample_hls_video[0] = 256


def test_video_copy(sample_hls_video):
    copied_video = sample_hls_video.copy()
    assert isinstance(copied_video, Video)
    assert not np.may_share_memory(copied_video._vid, sample_hls_video._vid)


def test_video_get_channel(sample_hls_video):
    channel = sample_hls_video.get_channel('h')
    assert isinstance(channel, Channel)
    channel = sample_hls_video.get_channel(0)
    assert isinstance(channel, Channel)
    with pytest.raises(ValueError):
        sample_hls_video.get_channel('unsupported')


def test_video_agg(sample_hls_video):
    result = sample_hls_video.agg('h', 'mean')
    assert isinstance(result, np.ndarray)
    assert result.shape == (sample_hls_video._vid.shape[0],)
    result = sample_hls_video.agg(1, lambda x: np.median(x, axis=1))
    assert isinstance(result, np.ndarray)
    assert result.shape == (sample_hls_video._vid.shape[0],)


def test_video_pct_change(sample_hls_video):
    result = sample_hls_video.pct_change('hue', 1)
    assert isinstance(result, np.ndarray)
    assert result.shape == (sample_hls_video._vid.shape[0],)


def test_video_mask(sample_hls_video):
    result = sample_hls_video.mask('h', 100, 200)
    assert isinstance(result, Video)
    assert not np.may_share_memory(result._vid, sample_hls_video._vid)
