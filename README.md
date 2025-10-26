CROWD DETECTION 

This project performs crowd detection and counting using the **YOLOv8 (You Only Look Once)** deep learning model.  
It analyzes video frames to identify people, count them, and generate annotated output videos with detection results.  
Ideal for monitoring crowd density in public places, events, or surveillance footage.


FEATURES
-  Detects and counts people in crowds using YOLOv8  
- Processes videos frame-by-frame with OpenCV  
- Supports adjustable confidence levels for detection accuracy  
-  Generates logs and confidence-level plots for analysis  
- Outputs annotated videos showing detected individuals  


TECH STACK
- Python 3
- YOLOv8 (Ultralytics)
- OpenCV– Video processing
- NumPy & Pandas – Data manipulation


FOLDER STRUCTURE
crowd-detection/
│
├── Input/                     # Folder for input videos or images
├── logs/                      # Logs and CSV outputs of detections
├── main.py                    # Main script for running detection
├── model_check.py             # Model testing or validation script
├── yolov8l.pt                 # YOLOv8 pre-trained model weights
├── output_video.mp4           # Sample processed output
├── output_detected_v8l.mp4    # Final annotated output video
├── confidence.jpeg            # showing confidence analysis
└── logs.zip                   # Compressed logs folder


SETUP INSTRUCTIONS
1. Install dependencies
   pip install ultralytics opencv-python numpy pandas 
2. Run the detection script
   python main.py


OUTPUT
* Annotated video with bounding boxes around detected people
* Log files containing frame-wise detections and confidence values
