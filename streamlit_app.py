from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import streamlit as st

from traffic_vision.camera import validate_camera_source
from traffic_vision.pipeline import PipelineConfig, TrafficPipeline
from traffic_vision.storage import SQLiteStore


st.set_page_config(page_title="Highway Vehicle Counter", layout="wide")
st.title("Highway Vehicle Counter")

with st.sidebar:
    source = st.text_input(
        "Camera source",
        value="0",
        help="Use 0 for the local webcam, a path like sample.mp4, or an HTTP/RTSP stream URL.",
    )
    db_path = st.text_input("SQLite database", value="traffic_counts.db")
    model_name = st.text_input("YOLO model", value="yolo26n.pt")
    confidence = st.slider("Confidence", 0.05, 0.95, 0.35, 0.05)
    all_classes = st.checkbox("Test mode: all YOLO objects")
    orientation = st.selectbox("Counting line", ("horizontal", "vertical"))
    line_position = st.number_input("Line position (px)", min_value=0, value=360, step=10)
    max_frames = st.number_input("Max frames per run", min_value=1, value=300, step=25)
    run = st.button("Start counting", type="primary")

store = SQLiteStore(Path(db_path))
store.initialize()

total_col, chart_col = st.columns([1, 2])
frame_placeholder = st.empty()
preview_caption = st.empty()
events_placeholder = st.empty()

def render_metrics() -> None:
    totals = store.totals_by_label()
    total_col.metric("Total vehicles", sum(totals.values()))
    if totals:
        chart_col.bar_chart(pd.DataFrame({"label": totals.keys(), "count": totals.values()}), x="label", y="count")
    recent = store.recent_events(limit=25)
    if recent:
        events_placeholder.dataframe(pd.DataFrame(recent), use_container_width=True, hide_index=True)


render_metrics()

if run:
    validation = validate_camera_source(source)
    if not validation.ok:
        st.error(validation.message)
        st.stop()

    st.sidebar.success(validation.message)
    try:
        pipeline = TrafficPipeline(
            PipelineConfig(
                source=validation.source,
                db_path=db_path,
                model_name=model_name,
                confidence=confidence,
                line_position=int(line_position),
                line_orientation=orientation,
                max_frames=int(max_frames),
                all_classes=all_classes,
            )
        )
        for snapshot in pipeline.run():
            frame = snapshot["frame"]
            frame_placeholder.image(frame, channels="BGR", use_container_width=True)
            preview_caption.caption(
                f"Frame {snapshot['frame_number']} | "
                f"Detections: {len(snapshot['detections'])} | "
                f"Tracks: {len(snapshot['tracks'])} | "
                f"Mode: {'all YOLO classes' if all_classes else 'vehicles only'}"
            )
            if snapshot["frame_number"] % 10 == 0 or snapshot["events"]:
                render_metrics()
            time.sleep(0.001)
        render_metrics()
    except RuntimeError as error:
        st.error(str(error))
