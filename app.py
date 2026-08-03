import streamlit as st
import cv2
import tempfile
import os
import time
import pandas as pd
import numpy as np
from detector import PPEDetector

# 1. Page Configuration
st.set_page_config(
    page_title="AI PPE Detection System",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS Styling for Premium Aesthetic
st.markdown("""
<style>
    /* Dark theme & Glassmorphism container styling */
    .stApp {
        background-color: #0e1117;
        color: #f1f5f9;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .metric-card h3 {
        margin: 0;
        font-size: 0.9rem;
        color: #94a3b8;
        font-weight: 500;
    }
    .metric-card p {
        margin: 8px 0 0 0;
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .alert-banner-danger {
        background-color: rgba(239, 68, 68, 0.2);
        border: 1px solid #ef4444;
        color: #fca5a5;
        border-radius: 8px;
        padding: 12px 16px;
        font-weight: 600;
        margin-bottom: 15px;
    }
    .alert-banner-success {
        background-color: rgba(34, 197, 94, 0.2);
        border: 1px solid #22c55e;
        color: #86efac;
        border-radius: 8px;
        padding: 12px 16px;
        font-weight: 600;
        margin-bottom: 15px;
    }

    /* Target all buttons for strong visibility */
    div.stButton > button, button[data-testid^="stBaseButton"] {
        border-radius: 8px !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.2s ease-in-out !important;
    }

    /* Standard / Secondary button styling (e.g. Stop Processing) */
    div.stButton > button:not([kind="primary"]), button[data-testid="stBaseButton-secondary"] {
        background-color: #334155 !important;
        color: #ffffff !important;
        border: 1px solid #64748b !important;
    }

    div.stButton > button:not([kind="primary"]):hover, button[data-testid="stBaseButton-secondary"]:hover {
        background-color: #dc2626 !important;
        color: #ffffff !important;
        border-color: #ef4444 !important;
        box-shadow: 0 4px 12px rgba(220, 38, 38, 0.4) !important;
    }

    /* Primary button styling (e.g. Start Processing) */
    div.stButton > button[kind="primary"], button[data-testid="stBaseButton-primary"] {
        background-color: #16a34a !important;
        color: #ffffff !important;
        border: 1px solid #22c55e !important;
    }

    div.stButton > button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover {
        background-color: #15803d !important;
        color: #ffffff !important;
        box-shadow: 0 4px 12px rgba(22, 163, 74, 0.4) !important;
    }

    /* Force bright white text & styled card backgrounds for st.metric */
    [data-testid="stMetric"] {
        background-color: #1e293b !important;
        border: 1px solid #475569 !important;
        border-radius: 10px !important;
        padding: 14px 18px !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3) !important;
    }
    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] > div, [data-testid="stMetricLabel"] label {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
    }
    [data-testid="stMetricValue"], [data-testid="stMetricValue"] > div {
        color: #ffffff !important;
        font-weight: 800 !important;
        font-size: 2.2rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load logs into DataFrame
def load_log_df(log_path="logs/ppe_log.txt"):
    if not os.path.exists(log_path):
        return pd.DataFrame(columns=["Timestamp", "Violation", "Confidence"])
    
    records = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if " - " in line:
                parts = line.split(" - ")
                if len(parts) >= 3:
                    ts = parts[0]
                    viol = parts[1]
                    conf = parts[2].replace("Confidence: ", "")
                    records.append({"Timestamp": ts, "Violation": viol, "Confidence": conf})
    return pd.DataFrame(records)

# 3. Sidebar Controls
st.sidebar.image("https://img.icons8.com/color/96/000000/safety-hat.png", width=70)
st.sidebar.title("🦺 PPE Detection")
st.sidebar.markdown("---")

st.sidebar.header("⚙️ Configuration")
input_option = st.sidebar.radio(
    "Select Video Source:",
    ("Demo Video", "Upload Video", "Live Webcam")
)

conf_threshold = st.sidebar.slider(
    "Detection Confidence Threshold",
    min_value=0.10, max_value=1.00, value=0.60, step=0.05
)

alert_interval = st.sidebar.slider(
    "Alert Cooldown (Seconds)",
    min_value=1.0, max_value=10.0, value=3.0, step=0.5
)

enable_sound = st.sidebar.checkbox("Enable Audio Alerts", value=True)

st.sidebar.markdown("---")
if st.sidebar.button("🧹 Clear Violation Logs"):
    if os.path.exists("logs/ppe_log.txt"):
        with open("logs/ppe_log.txt", "w", encoding="utf-8") as f:
            f.write("")
        st.sidebar.success("Log cleared!")
        st.rerun()

# 4. Main Interface Header
col_header, col_status = st.columns([3, 1])
with col_header:
    st.title("🦺 Real-Time Construction Safety PPE Monitor")
    st.markdown("Automated Computer Vision Safety Compliance Monitoring powered by **YOLOv8**.")

with col_status:
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("System Ready")

# Tabs Layout
tab_live, tab_video, tab_logs, tab_analytics = st.tabs([
    "📹 Live Detection Feed", 
    "🎬 Processed Output Video", 
    "📄 Violation Logs", 
    "📊 Analytics & Report"
])

# Initialize Detector
detector = PPEDetector(
    model_path="models/best.pt",
    log_path="logs/ppe_log.txt",
    sound_path="sounds/alert.wav",
    conf_threshold=conf_threshold,
    alert_interval=alert_interval,
    enable_sound=enable_sound
)

# --------------------------
# TAB 1: Live Detection Feed
# --------------------------
with tab_live:
    # Source Resolution
    video_source = None
    temp_file_path = None

    if input_option == "Demo Video":
        demo_path = "videos/construction.mp4"
        if os.path.exists(demo_path):
            video_source = demo_path
        else:
            st.error(f"Demo video file not found at '{demo_path}'. Please check file path.")

    elif input_option == "Upload Video":
        uploaded_file = st.file_uploader("Upload Video File (MP4, AVI, MOV)", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            video_source = tfile.name
            temp_file_path = tfile.name

    elif input_option == "Live Webcam":
        video_source = 0

    st.markdown("### Control & Metrics")
    
    # Static Button Controls (Fixed Position - Never Shifts)
    btn_col1, btn_col2, _ = st.columns([1, 1, 2])
    with btn_col1:
        start_button = st.button("▶️ Start Processing", type="primary", use_container_width=True)
    with btn_col2:
        stop_button = st.button("⏹️ Stop Processing", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # KPI Metric Cards Placeholders
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        metric_fps = st.empty()
    with m_col2:
        metric_helmet = st.empty()
    with m_col3:
        metric_vest = st.empty()
    with m_col4:
        metric_mask = st.empty()

    metric_fps.metric("Processing FPS", "0.0")
    metric_helmet.metric("Helmet Violations", "0", delta_color="inverse")
    metric_vest.metric("Vest Violations", "0", delta_color="inverse")
    metric_mask.metric("Mask Violations", "0", delta_color="inverse")

    st.markdown("---")
    
    col_video, col_info = st.columns([3, 1])
    
    with col_video:
        st.markdown("#### Live Bounding Box Stream")
        frame_placeholder = st.empty()

    with col_info:
        st.markdown("#### System Status")
        status_box = st.empty()
        status_box.markdown("<div class='alert-banner-success'>🟢 Monitoring Active & Safe</div>", unsafe_allow_html=True)
        
        st.markdown("#### Recent Live Logs")
        live_log_placeholder = st.empty()

        st.markdown("#### Active Warning Alert")
        banner_placeholder = st.empty()
        banner_placeholder.markdown("<div class='alert-banner-success'>✅ No Active Warnings</div>", unsafe_allow_html=True)

    if start_button:
        if video_source is None:
            st.warning("Please upload or select a valid video source before starting.")
        else:
            cap = cv2.VideoCapture(video_source)
            if not cap.isOpened():
                st.error("Unable to open the selected video source.")
            else:
                detector.reset_stats()
                
                # Output video setup
                output_dir = "output"
                os.makedirs(output_dir, exist_ok=True)
                out_path = os.path.join(output_dir, "detected_construction.mp4")
                
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
                fps_input = cap.get(cv2.CAP_PROP_FPS) or 25.0
                
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_writer = cv2.VideoWriter(out_path, fourcc, fps_input, (width, height))

                st.toast("Detection started!", icon="🚀")

                while cap.isOpened() and not stop_button:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    processed_frame, info = detector.process_frame(frame, draw_hud=False)

                    # Write to output video file
                    out_writer.write(processed_frame)

                    # Convert BGR (OpenCV) to RGB (Streamlit)
                    rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

                    # Update Metrics
                    metric_fps.metric("Processing FPS", f"{info['fps']:.1f}")
                    metric_helmet.metric("Helmet Violations", f"{info['helmet_count']}")
                    metric_vest.metric("Vest Violations", f"{info['vest_count']}")
                    metric_mask.metric("Mask Violations", f"{info['mask_count']}")

                    # Update Warning Banner (Right side panel below logs)
                    if info["warning"]:
                        banner_placeholder.markdown(
                            f"<div class='alert-banner-danger'>⚠️ <b>{info['warning']}</b></div>",
                            unsafe_allow_html=True
                        )
                        status_box.markdown(
                            f"<div class='alert-banner-danger'>🔴 VIOLATION DETECTED</div>",
                            unsafe_allow_html=True
                        )
                    else:
                        banner_placeholder.markdown(
                            "<div class='alert-banner-success'>✅ No Active Warnings</div>",
                            unsafe_allow_html=True
                        )
                        status_box.markdown(
                            "<div class='alert-banner-success'>🟢 Monitoring Active & Safe</div>",
                            unsafe_allow_html=True
                        )

                    # Live Log View
                    df_log = load_log_df("logs/ppe_log.txt")
                    if not df_log.empty:
                        live_log_placeholder.dataframe(df_log.tail(5), use_container_width=True)

                cap.release()
                out_writer.release()
                st.success("Video processing completed successfully!")
                st.toast("Saved output video to output/detected_construction.mp4", icon="✅")

# --------------------------------
# TAB 2: Processed Output Video
# --------------------------------
with tab_video:
    st.markdown("### Processed Detection Video")
    output_video_file = "output/detected_construction.mp4"
    
    if os.path.exists(output_video_file) and os.path.getsize(output_video_file) > 0:
        st.success("Processed video is available for playback and download.")
        
        with open(output_video_file, "rb") as vf:
            video_bytes = vf.read()
            st.video(video_bytes)
            
            st.download_button(
                label="📥 Download Processed Video (.mp4)",
                data=video_bytes,
                file_name="ppe_detected_output.mp4",
                mime="video/mp4",
                type="primary"
            )
    else:
        st.info("No processed video available yet. Please run the detection on a video stream in the 'Live Detection Feed' tab.")

# ----------------------
# TAB 3: Violation Logs
# ----------------------
with tab_logs:
    st.markdown("### PPE Violation Audit Logs")
    df_logs = load_log_df("logs/ppe_log.txt")
    
    if not df_logs.empty:
        col_log_count, col_log_down = st.columns([3, 1])
        with col_log_count:
            st.write(f"Total Logged Violations: **{len(df_logs)}**")
        with col_log_down:
            csv_data = df_logs.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export Logs (CSV)",
                data=csv_data,
                file_name="ppe_violation_logs.csv",
                mime="text/csv"
            )

        st.dataframe(df_logs, use_container_width=True)
    else:
        st.info("No violations logged yet.")

# --------------------------
# TAB 4: Analytics & Report
# --------------------------
with tab_analytics:
    st.markdown("### Violation Analytics & Breakdown")
    df_analytics = load_log_df("logs/ppe_log.txt")
    
    if not df_analytics.empty:
        counts = df_analytics["Violation"].value_counts()
        
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.markdown("#### Violations by Category")
            st.bar_chart(counts)

        with chart_col2:
            st.markdown("#### Summary Table")
            summary_df = counts.reset_index()
            summary_df.columns = ["Violation Category", "Total Count"]
            st.dataframe(summary_df, use_container_width=True)
    else:
        st.info("Run detections to generate analytics and violation summary charts.")