import cv2
import time
import os
import threading
from datetime import datetime
from ultralytics import YOLO

class PPEDetector:
    def __init__(self, model_path="models/best.pt", log_path="logs/ppe_log.txt", sound_path="sounds/alert.wav", conf_threshold=0.60, alert_interval=3.0, enable_sound=True):
        self.model_path = model_path
        self.log_path = log_path
        self.sound_path = sound_path
        self.conf_threshold = conf_threshold
        self.alert_interval = alert_interval
        self.enable_sound = enable_sound

        self.model = YOLO(model_path)
        
        # Ensure log directory exists
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        
        self.reset_stats()

    def reset_stats(self):
        self.helmet_count = 0
        self.vest_count = 0
        self.mask_count = 0

        self.last_logged = {
            "Helmet": 0.0,
            "Vest": 0.0,
            "Mask": 0.0,
        }
        self.last_alert_time = 0.0
        self.prev_time = time.time()
        self.fps = 0.0

    def play_alert_sound(self):
        if not self.enable_sound:
            return
        try:
            import platform
            if platform.system() != "Windows":
                # Audio hardware does not exist on cloud servers (e.g. Streamlit Cloud Linux container)
                return
            from playsound import playsound
            if self.sound_path and os.path.exists(self.sound_path):
                playsound(self.sound_path)
        except Exception:
            # Sound hardware unavailable in headless cloud environments (e.g. Streamlit Cloud)
            pass

    def log_violation(self, violation_type, conf):
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(f"{datetime.now()} - {violation_type} - Confidence: {conf:.2f}\n")
        except Exception as e:
            print(f"Logging error: {e}")

    def process_frame(self, frame, draw_hud=True):
        current_time = time.time()
        self.fps = 1 / max(current_time - self.prev_time, 1e-6)
        self.prev_time = current_time

        warning = ""
        results = self.model(frame, verbose=False, imgsz=640)

        frame_violations = []

        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if conf < self.conf_threshold:
                    continue

                label = self.model.names[cls]
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                color = (0, 255, 0) # Green for compliant PPE

                if label == "NO-Hardhat":
                    color = (0, 0, 255)
                    warning = "WARNING : HELMET MISSING"
                    frame_violations.append("NO-Hardhat")

                    if current_time - self.last_logged["Helmet"] > 3.0:
                        self.helmet_count += 1
                        self.log_violation("Helmet Missing", conf)
                        self.last_logged["Helmet"] = current_time

                elif label == "NO-Safety Vest":
                    color = (0, 0, 255)
                    warning = "WARNING : SAFETY VEST MISSING"
                    frame_violations.append("NO-Safety Vest")

                    if current_time - self.last_logged["Vest"] > 3.0:
                        self.vest_count += 1
                        self.log_violation("Safety Vest Missing", conf)
                        self.last_logged["Vest"] = current_time

                elif label == "NO-Mask":
                    color = (0, 0, 255)
                    warning = "WARNING : MASK MISSING"
                    frame_violations.append("NO-Mask")

                    if current_time - self.last_logged["Mask"] > 3.0:
                        self.mask_count += 1
                        self.log_violation("Mask Missing", conf)
                        self.last_logged["Mask"] = current_time

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Trigger sound alert if warning exists and interval passed
        if warning and (current_time - self.last_alert_time > self.alert_interval):
            threading.Thread(target=self.play_alert_sound, daemon=True).start()
            self.last_alert_time = current_time

        # Draw warning box at bottom-right corner of the frame
        if warning:
            h, w, _ = frame.shape
            box_w, box_h = 420, 55
            bx2 = w - 15
            by2 = h - 15
            bx1 = max(bx2 - box_w, 10)
            by1 = max(by2 - box_h, 10)

            # Draw solid red warning box with white border
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 0, 255), -1)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 255, 255), 2)
            cv2.putText(frame, warning, (bx1 + 15, by1 + 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        # Draw HUD overlay if requested
        if draw_hud:
            cv2.rectangle(frame, (10, 10), (320, 140), (40, 40, 40), -1)
            cv2.putText(frame, f"FPS : {self.fps:.2f}", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Helmet : {self.helmet_count}", (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Vest : {self.vest_count}", (20, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Mask : {self.mask_count}", (20, 125),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return frame, {
            "warning": warning,
            "violations": frame_violations,
            "fps": self.fps,
            "helmet_count": self.helmet_count,
            "vest_count": self.vest_count,
            "mask_count": self.mask_count,
        }
