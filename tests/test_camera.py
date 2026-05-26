from traffic_vision.camera import is_stream_url, parse_camera_source, validate_camera_source


def test_parse_camera_source_accepts_webcam_file_and_url() -> None:
    assert parse_camera_source("0") == 0
    assert parse_camera_source(0) == 0
    assert parse_camera_source("sample.mp4") == "sample.mp4"
    assert parse_camera_source("rtsp://example.com/live") == "rtsp://example.com/live"


def test_is_stream_url_for_http_and_rtsp_sources() -> None:
    assert is_stream_url("http://example.com/video")
    assert is_stream_url("https://example.com/video")
    assert is_stream_url("rtsp://example.com/live")
    assert not is_stream_url("sample.mp4")


def test_validate_camera_source_reports_missing_video_file() -> None:
    result = validate_camera_source("definitely_missing_sample.mp4")

    assert not result.ok
    assert "does not exist" in result.message
