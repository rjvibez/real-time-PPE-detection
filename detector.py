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

                # Define compliance color and human-friendly display label
                if label == "NO-Hardhat":
                    color = (40, 40, 230) # Crimson Red for violation
                    display_label = f"NO HELMET {conf*100:.0f}%"
                    warning = "WARNING : HELMET MISSING"
                    frame_violations.append("NO-Hardhat")

                    if current_time - self.last_logged["Helmet"] > 3.0:
                        self.helmet_count += 1
                        self.log_violation("Helmet Missing", conf)
                        self.last_logged["Helmet"] = current_time

                elif label == "NO-Safety Vest":
                    color = (40, 40, 230) # Crimson Red for violation
                    display_label = f"NO VEST {conf*100:.0f}%"
                    warning = "WARNING : SAFETY VEST MISSING"
                    frame_violations.append("NO-Safety Vest")

                    if current_time - self.last_logged["Vest"] > 3.0:
                        self.vest_count += 1
                        self.log_violation("Safety Vest Missing", conf)
                        self.last_logged["Vest"] = current_time

                elif label == "NO-Mask":
                    color = (40, 40, 230) # Crimson Red for violation
                    display_label = f"NO MASK {conf*100:.0f}%"
                    warning = "WARNING : MASK MISSING"
                    frame_violations.append("NO-Mask")

                    if current_time - self.last_logged["Mask"] > 3.0:
                        self.mask_count += 1
                        self.log_violation("Mask Missing", conf)
                        self.last_logged["Mask"] = current_time

                elif label in ["Hardhat", "Safety-Helmet", "Helmet"]:
                    color = (40, 200, 40) # Neon Green for compliance
                    display_label = f"HELMET {conf*100:.0f}%"
                elif label in ["Safety Vest", "Vest"]:
                    color = (40, 200, 40) # Neon Green for compliance
                    display_label = f"SAFETY VEST {conf*100:.0f}%"
                elif label in ["Mask", "Face-Mask"]:
                    color = (40, 200, 40) # Neon Green for compliance
                    display_label = f"MASK {conf*100:.0f}%"
                else:
                    color = (40, 200, 40) if "NO-" not in label else (40, 40, 230)
                    clean_name = label.replace("NO-", "NO ").replace("-", " ").upper()
                    display_label = f"{clean_name} {conf*100:.0f}%"

                # 1. Main Bounding Box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

                # 2. High-Tech Corner Accent Brackets
                c_len = min(14, max(4, int((x2 - x1) / 4), int((y2 - y1) / 4)))
                cv2.line(frame, (x1, y1), (x1 + c_len, y1), (255, 255, 255), 2, cv2.LINE_AA)
                cv2.line(frame, (x1, y1), (x1, y1 + c_len), (255, 255, 255), 2, cv2.LINE_AA)
                cv2.line(frame, (x2, y1), (x2 - c_len, y1), (255, 255, 255), 2, cv2.LINE_AA)
                cv2.line(frame, (x2, y1), (x2, y1 + c_len), (255, 255, 255), 2, cv2.LINE_AA)
                cv2.line(frame, (x1, y2), (x1 + c_len, y2), (255, 255, 255), 2, cv2.LINE_AA)
                cv2.line(frame, (x1, y2), (x1, y2 - c_len), (255, 255, 255), 2, cv2.LINE_AA)
                cv2.line(frame, (x2, y2), (x2 - c_len, y2), (255, 255, 255), 2, cv2.LINE_AA)
                cv2.line(frame, (x2, y2), (x2, y2 - c_len), (255, 255, 255), 2, cv2.LINE_AA)

                # 3. Solid Filled Label Badge Background with White Text
                font_scale = 0.45
                font_thick = 1
                (tw, th), _ = cv2.getTextSize(display_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)

                bg_y1 = max(y1 - th - 10, 0)
                bg_y2 = bg_y1 + th + 10
                bg_x1 = x1
                bg_x2 = x1 + tw + 12

                cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
                cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, display_label, (bg_x1 + 6, bg_y2 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)

        # Trigger sound alert if warning exists and interval passed
        if warning and (current_time - self.last_alert_time > self.alert_interval):
            threading.Thread(target=self.play_alert_sound, daemon=True).start()
            self.last_alert_time = current_time

        # Draw warning box at bottom-right corner of the frame
        if warning:
            h, w, _ = frame.shape
            font_scale = max(0.45, w / 1200.0)
            (tw, th), _ = cv2.getTextSize(warning, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
            pad_x, pad_y = 12, 10
            bx2 = w - 10
            by2 = h - 10
            bx1 = max(bx2 - tw - pad_x * 2, 10)
            by1 = max(by2 - th - pad_y * 2, 10)

            # Draw solid crimson red warning banner with white border
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (40, 40, 230), -1)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, warning, (bx1 + pad_x, by2 - pad_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2, cv2.LINE_AA)

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
