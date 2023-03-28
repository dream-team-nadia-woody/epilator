from video.vid import Video
import pytest
import os

@pytest.fixture
def test_video():
    current_dir = os.path.dirname(__file__)
    