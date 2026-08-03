# 🦺 Real-Time PPE Detection System for Construction Site Safety

## 📌 Project Overview

This project is an **AI-powered Personal Protective Equipment (PPE) Detection System** developed using Python, OpenCV, Ultralytics YOLOv8, and Streamlit. It monitors construction workers from video feeds or webcams and automatically detects whether workers are wearing essential safety equipment such as **helmets, safety vests, and masks**.

If a worker is detected without required PPE, the system:
- 🚨 **Displays a visual warning** in a dedicated bottom-right warning box on the video frame.
- 🔊 **Plays an audio alert** to notify safety supervisors.
- 📄 **Logs the violation** with timestamps, violation categories, and confidence scores.
- 🎬 **Saves the annotated video** to disk for audit and download.
- 📊 **Displays live KPI analytics** on an interactive Streamlit web dashboard.

---

## ⚙️ How It Works (System Architecture & Pipeline)

The system works through an end-to-end Computer Vision and Deep Learning pipeline:

```
[ Video Input / Webcam ] ➡️ [ OpenCV Frame Extraction ] ➡️ [ YOLOv8 Inference (best.pt) ]
                                                                      │
                                                                      ▼
[ Saved Output MP4 ] ⬅️ [ Streamlit / CLI Dashboard ] ⬅️ [ Bounding Box & Alert Overlay ]
                                                                      │
                                                                      ▼
                                                       [ Log File & Audio Alerts ]
```

### 1. Input Acquisition
- OpenCV (`cv2.VideoCapture`) captures video frames frame-by-frame from three supported sources:
  - 📁 **Uploaded Video Files** (`.mp4`, `.avi`, `.mov`)
  - 🎬 **Sample Construction Video** (`videos/construction.mp4`)
  - 📷 **Live Webcam Feed**

### 2. YOLOv8 Deep Learning Inference
- Each frame is passed to the fine-tuned custom YOLOv8 object detection model (`models/best.pt`).
- The model detects 6 distinct classes:
  - **Compliant Classes**: `Hardhat`, `Safety Vest`, `Mask`
  - **Non-Compliant Classes**: `NO-Hardhat`, `NO-Safety Vest`, `NO-Mask`
- Detections are filtered based on the configurable **Confidence Threshold** (default: `0.60` or user-tuned via slider).

### 3. Bounding Box & Visual Alert Generation
- **Compliant PPE**: Rendered with **Green Bounding Boxes** `(0, 255, 0)` and class confidence scores.
- **Non-Compliant PPE**: Rendered with **Red Bounding Boxes** `(0, 0, 255)` and class labels.
- **Bottom-Right Warning Box**: If a violation is detected (e.g., `NO-Mask`), a solid red warning box with a white outline is dynamically drawn at the **bottom-right corner of the video frame** displaying `WARNING : MASK MISSING`.

### 4. Asynchronous Audio Alert & Smart Logging
- **Audio Alerts**: When a violation occurs, a background thread triggers `sounds/alert.wav` via `playsound` without blocking video processing. Audio alerts respect a configurable **Alert Cooldown Interval** (default: 3.0 seconds).
- **Smart Logging**: Violations are written to `logs/ppe_log.txt` in real time:
  ```
  2026-08-03 08:20:15.123456 - Mask Missing - Confidence: 0.85
  ```
  *Note: Logging uses a 3-second throttle per category to prevent redundant log spam.*

### 5. Streamlit Dashboard & Web Deployment (`app.py`)
- **Fixed Button Controls**: Start (`▶️`) and Stop (`⏹️`) processing buttons remain static and never shift layout during alerts.
- **High-Contrast KPI Metrics**: Displays **Processing FPS**, **Helmet Violations**, **Vest Violations**, and **Mask Violations** in bright white text on dark cards.
- **Right-Side Alert Panel**: Displays active warning notifications directly below the live log table.
- **Video Export**: Processes and encodes annotated frames into `output/detected_construction.mp4`, complete with an in-browser player and direct download button.
- **Analytics & Audit**: Parses logs into interactive Pandas DataFrames, offering CSV downloads and category breakdown bar charts.

---

## 🚀 Key Features

- 🧠 **YOLOv8 Neural Network**: High-speed real-time object detection.
- 🦺 **Comprehensive PPE Coverage**: Monitors helmets, safety vests, and masks.
- 🟩 **Green / Red Visual Classification**: Instant visual feedback for safety compliance.
- 📍 **Bottom-Right Warning Sign**: Clear, non-intrusive warning box on the video frame.
- 🔊 **Audio Alerts**: Audible warning system for non-compliant workers.
- 📈 **Real-Time KPI Cards**: High-contrast, bright white metric displays for FPS and violation counts.
- 📄 **Violation Audit Logs**: Automated timestamped logging with CSV export.
- 🎬 **Video Player & Downloader**: Export processed detection videos directly from Streamlit.
- 📊 **Visual Analytics**: Interactive violation breakdown charts.

---

## 🛠️ Technologies Used

- **Python 3.x**
- **OpenCV (`cv2`)**: Video capture, frame manipulation, drawing overlays, and video writing.
- **Ultralytics YOLOv8**: Deep learning object detection architecture.
- **Streamlit**: Web deployment dashboard framework.
- **Pandas**: Data parsing and analytics visualization.
- **Playsound**: Threaded audio alert system.
- **NumPy**: Matrix operations on frame arrays.

---

## 📁 Project Structure

```
Final-project/
│
├── models/
│   └── best.pt                # Fine-tuned YOLOv8 custom weights model
│
├── videos/
│   └── construction.mp4       # Demo construction site video
│
├── sounds/
│   └── alert.wav              # Audio alert sound file
│
├── logs/
│   └── ppe_log.txt            # Real-time timestamped violation logs
│
├── output/
│   └── detected_construction.mp4  # Output video with bounding boxes & warnings
│
├── detector.py                # Core PPEDetector modular engine
├── main.py                    # Command-line interface (CLI) entry point
├── app.py                     # Streamlit web application dashboard
├── requirements.txt           # Required Python packages
└── README.md                  # Detailed project documentation
```

---

## ⚙️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Final-project
   ```

2. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Running the Project

### Option A: Run via Streamlit Web Application (Recommended)

Launch the interactive web application:

```bash
streamlit run app.py
```

The web dashboard will automatically open in your browser at `http://localhost:8501`.

#### How to Use the Web App:
1. Select a **Video Source** from the sidebar (Demo Video, Upload Video, or Live Webcam).
2. Adjust the **Confidence Threshold** slider if needed (default `0.60`).
3. Click **▶️ Start Processing** to launch detection.
4. View real-time bounding boxes, bottom-right warning sign, and KPI metrics.
5. Click **⏹️ Stop Processing** at any time.
6. Switch to the **Processed Output Video** tab to watch or download the output `.mp4` video.
7. Check the **Violation Logs** tab to export CSV audit reports.

---

### Option B: Run via Command Line Interface (CLI)

Run detection directly in OpenCV desktop window:

```bash
python main.py
```

To run on a custom video file via CLI:

```bash
python main.py path/to/your_video.mp4
```

*Press `q` on the keyboard to exit the CLI window.*

---

## 📄 Violation Log Format

Detected violations are appended to `logs/ppe_log.txt`:

```text
2026-08-03 08:15:30.451234 - Helmet Missing - Confidence: 0.88
2026-08-03 08:15:34.112849 - Safety Vest Missing - Confidence: 0.91
2026-08-03 08:15:40.890123 - Mask Missing - Confidence: 0.84
```

Each log line includes:
- **Timestamp**: Exact date and time of detection.
- **Violation Category**: `Helmet Missing`, `Safety Vest Missing`, or `Mask Missing`.
- **Confidence**: Model detection confidence score (0.00 to 1.00).

---

## 🎯 Future Improvements

- 📹 **Multi-Camera CCTV Integration**: Support RTSP/IP camera streams simultaneously.
- 👤 **Worker Identification**: Integrate face recognition or ID badge scanning.
- ☁️ **Cloud Database Sync**: Send logs directly to PostgreSQL / AWS S3.
- 📱 **Automated Notifications**: Send instant Telegram/Email alerts on critical safety violations.

---

## 👨‍💻 Author

Developed by **Rajesh**
