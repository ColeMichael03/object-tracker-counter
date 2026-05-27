# Highway Vehicle Counter

A Python computer vision app for detecting, tracking, and counting vehicles in a highway camera feed. It uses OpenCV for camera/video input, Ultralytics YOLO for object detection, lightweight centroid tracking for stable object IDs, SQLite for event storage, and Streamlit for a dashboard.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` uses `opencv-python-headless` so the app can deploy on Streamlit Cloud, which does not provide desktop GUI libraries.

## Run The Dashboard

```bash
streamlit run streamlit_app.py
```

The dashboard supports three input modes:

```text
Video upload
rtsp://user:password@192.168.1.50:554/stream1
http://192.168.1.50/video
Local webcam source 0
```

On Streamlit Cloud, use **Video upload** or **Stream URL**. `Local webcam source 0` is only for running the app on your own machine because Streamlit Cloud cannot access your laptop webcam.

OpenCV validates the source before the YOLO model starts. If the source cannot be opened, the dashboard shows whether the problem looks like a missing file, unavailable webcam, or unreachable HTTP/RTSP stream.

Enable **Test mode: all YOLO objects** in the sidebar to detect, track, draw, and count every YOLO class. Leave it off for normal traffic counting, which filters to `car`, `truck`, `bus`, and `motorcycle`.

## Run The Counter Headlessly

```bash
python -m traffic_vision.pipeline --source 0 --db traffic_counts.db
```

More source examples:

```bash
python -m traffic_vision.pipeline --source sample.mp4 --db traffic_counts.db
python -m traffic_vision.pipeline --source rtsp://user:password@192.168.1.50:554/stream1
python -m traffic_vision.pipeline --source http://192.168.1.50/video --all-classes
```

## Test

```bash
python -m pytest
```

The unit tests cover the tracking, line-crossing counter, and SQLite storage layers without requiring a camera or YOLO model download. The default detector model is `yolo26n.pt`, which Ultralytics lists as its current small pretrained detection model family.
