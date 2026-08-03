from datetime import datetime
from playsound import playsound
import threading


def play_alert():
    playsound("sounds/alert.wav")


def write_log(log_file, violation, confidence):
    log_file.write(
        f"{datetime.now()} - {violation} - Confidence: {confidence:.2f}\n"
    )
    log_file.flush()


def play_alert_thread():
    threading.Thread(target=play_alert, daemon=True).start()