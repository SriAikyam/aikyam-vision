import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import json

from vision.video_processor import extract, VideoMetadata, VideoStream, AudioStream

@pytest.fixture
def mock_ffprobe_success():
    return {
        "format": {
            "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
            "duration": "12.34",
            "size": "56789",
            "bit_rate": "36800",
            "tags": {
                "creation_time": "2026-07-10T12:00:00.000000Z",
                "location": "+28.6139+077.2090/"
            }
        },
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1920,
                "height": 1080,
                "r_frame_rate": "30/1",
                "pix_fmt": "yuv420p",
                "bit_rate": "30000",
                "tags": {
                    "creation_time": "2026-07-10T12:00:00.000000Z"
                }
            },
            {
                "codec_type": "audio",
                "codec_name": "aac",
                "sample_rate": "44100",
                "channels": 2,
                "bit_rate": "6800"
            }
        ]
    }

@pytest.fixture
def mock_ffprobe_na():
    # Simulate a case where ffprobe returns "N/A" for some values
    return {
        "format": {
            "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
            "duration": "N/A",
            "size": "N/A",
            "bit_rate": "N/A",
            "tags": {}
        },
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1920,
                "height": 1080,
                "r_frame_rate": "30/1",
                "pix_fmt": "yuv420p",
                "bit_rate": "N/A"
            },
            {
                "codec_type": "audio",
                "codec_name": "aac",
                "sample_rate": "N/A",
                "channels": 2,
                "bit_rate": "N/A"
            }
        ]
    }

@patch("subprocess.run")
def test_extract_success(mock_run, mock_ffprobe_success):
    # Mock ffprobe run
    mock_ffprobe_res = MagicMock()
    mock_ffprobe_res.returncode = 0
    mock_ffprobe_res.stdout = json.dumps(mock_ffprobe_success)
    
    # Mock ffmpeg frames and audio extraction
    mock_ffmpeg_res = MagicMock()
    mock_ffmpeg_res.returncode = 0
    
    mock_run.side_effect = [mock_ffprobe_res, mock_ffmpeg_res, mock_ffmpeg_res, mock_ffmpeg_res, mock_ffmpeg_res, mock_ffmpeg_res, mock_ffmpeg_res]

    with patch("pathlib.Path.exists", return_value=True):
        meta = extract("dummy.mp4")
        
        assert meta.error is None
        assert meta.format_name == "mov,mp4,m4a,3gp,3g2,mj2"
        assert meta.duration_seconds == 12.34
        assert meta.size_bytes == 56789
        assert meta.overall_bit_rate_kbps == 36
        assert meta.gps_lat == 28.6139
        assert meta.gps_lon == 77.2090
        assert meta.video.codec == "h264"
        assert meta.video.width == 1920
        assert meta.video.height == 1080
        assert meta.video.fps == 30.0
        assert meta.audio.codec == "aac"
        assert meta.audio.sample_rate_hz == 44100

@patch("subprocess.run")
def test_extract_na_values(mock_run, mock_ffprobe_na):
    # Mock ffprobe run
    mock_ffprobe_res = MagicMock()
    mock_ffprobe_res.returncode = 0
    mock_ffprobe_res.stdout = json.dumps(mock_ffprobe_na)
    
    mock_run.return_value = mock_ffprobe_res
    
    # We expect extract not to crash when encountering "N/A"
    meta = extract("dummy.mp4")
    assert meta.error is not None # Duration <= 0 because it's N/A (0.0)
    assert meta.duration_seconds == 0.0

@patch("subprocess.run")
def test_ffmpeg_missing_handled_gracefully(mock_run, mock_ffprobe_success):
    # Mock ffprobe run succeeds, but ffmpeg raises FileNotFoundError (e.g. not installed)
    mock_ffprobe_res = MagicMock()
    mock_ffprobe_res.returncode = 0
    mock_ffprobe_res.stdout = json.dumps(mock_ffprobe_success)
    
    def side_effect(cmd, *args, **kwargs):
        if cmd[0] == "ffprobe":
            return mock_ffprobe_res
        else:
            raise FileNotFoundError("ffmpeg not found")
            
    mock_run.side_effect = side_effect
    
    # We expect extract NOT to raise FileNotFoundError now, but instead return metadata with an error set
    meta = extract("dummy.mp4")
    assert meta.error is not None
    assert "ffmpeg" in meta.error
    assert "not found" in meta.error
