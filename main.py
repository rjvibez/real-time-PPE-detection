import sys
import cv2
from detector import PPEDetector

def main():
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = "videos/construction.mp4"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video at {video_path}.")
        sys.exit(1)

    detector = PPEDetector(
        model_path="models/best.pt",
        log_path="logs/ppe_log.txt",
        sound_path="sounds/alert.wav",
        conf_threshold=0.60
    )

    print("Starting Real-Time PPE Detection System... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame, info = detector.process_frame(frame, draw_hud=True)

        cv2.imshow("Real-Time PPE Detection", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("PPE Detection Stopped.")

if __name__ == "__main__":
    main()
