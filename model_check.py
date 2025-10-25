import cv2
from ultralytics import YOLO
import os
import csv
import time
import logging


def crowd_detection(input_video, output_video, confidence=0.7, show_preview=False):
    # === Create logs directory ===
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # === Set up file names with timestamps ===
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f"crowd_detection_{timestamp}.log")
    csv_file = os.path.join(log_dir, f"detection_log_{timestamp}.csv")

    # === Configure logging ===
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("==============================================================")
    logger.info("Starting YOLOv8 Crowd Detection with Logging Enabled")
    logger.info(f"Confidence threshold: {confidence}")
    logger.info("==============================================================")

    # === Initialize YOLOv8 Model ===
    logger.info("Loading YOLOv8 model (auto-download if not present)...")
    model = YOLO('yolov8l.pt')

    # === Open video ===
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        logger.error("Error: Could not open video file!")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Video Info: {width}x{height} @ {fps} FPS | Total frames: {total_frames}")

    # === Output video setup ===
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # === CSV Setup ===
    with open(csv_file, 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["Timestamp", "Frame", "Class", "Confidence", "People Count"])

    frame_count = 0
    person_class_id = 0  # COCO class ID for 'person'
    logger.info("Processing frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        current_time = time.strftime('%Y-%m-%d %H:%M:%S')

        results = model(frame, conf=confidence, verbose=False)
        boxes = results[0].boxes
        person_count = 0

        # Detection Loop
        with open(csv_file, 'a', newline='') as f:
            csv_writer = csv.writer(f)
            for box in boxes:
                cls = int(box.cls[0])
                conf_val = float(box.conf[0])
                class_name = model.names[cls]

                # Count people
                if cls == person_class_id:
                    person_count += 1

                # Log detection to CSV
                csv_writer.writerow([current_time, frame_count, class_name, f'{conf_val:.2f}', person_count])

                # Draw bounding boxes
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name} {conf_val:.2f}"
                cv2.putText(frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Overlay people count
        cv2.rectangle(frame, (10, 10), (350, 50), (0, 0, 0), -1)
        cv2.putText(frame, f"People Detected: {person_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Write frame
        out.write(frame)

        # Optional preview
        if show_preview:
            cv2.imshow('Crowd Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.warning("User stopped the process manually.")
                break

        # Progress update
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            logger.info(f"Progress: {frame_count}/{total_frames} ({progress:.1f}%) | People Detected: {person_count}")

    # === Cleanup ===
    cap.release()
    out.release()
    if show_preview:
        cv2.destroyAllWindows()

    logger.info("==============================================================")
    logger.info("Processing Completed Successfully!")
    logger.info(f"Output Video: {output_video}")
    logger.info(f"Log File: {log_file}")
    logger.info(f"CSV Summary: {csv_file}")
    logger.info(f"Total Frames Processed: {frame_count}")
    logger.info("==============================================================")


if __name__ == "__main__":
    # Modify these paths as needed
    input_video_path = r"/Users/anubhi/Desktop/video/output_video.mp4"
    output_video_path = r"/Users/anubhi/Desktop/video/output_detected_v8l.mp4"

    # Run detection
    crowd_detection(
        input_video=input_video_path,
        output_video=output_video_path,
        confidence=0.4,
        show_preview=False
    )

    print("\n" + "="*60)
    print("TIP: Adjust confidence parameter for more/fewer detections")
    print("  - Lower (0.3): More detections (may include false positives)")
    print("  - Higher (0.7): Fewer but highly confident detections")
    print("="*60)
