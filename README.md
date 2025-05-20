# :camera: Video Frame Extractor & YOLO Pre-Labeler

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)
![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

Welcome to the **Video Frame Extractor & YOLO Pre-Labeler**! This Python script automates frame extraction, person detection with YOLOv8, and Pascal VOC XML annotation generation. :rocket:

## :wrench: Features

- :movie_camera: **Frame Extraction**: Extracts frames at a specified interval.
- :detective: **Object Detection**: Detects people using YOLOv8.
- :memo: **Annotation Generation**: Converts detections to Pascal VOC XML.
- :mag: **Manual Verification**: Guides you to use imgLabeler for refining annotations.

## :clipboard: Prerequisites

- :snake: Python 3.8+
- :package: Install dependencies:
  ```bash
  pip install opencv-python ultralytics

:camera_flash: OpenCV for frame extraction

:brain: YOLOv8 model weights (auto-downloaded by ultralytics)

:pushpin: imgLabeler (optional, for manual annotation refinement)

:rocket: Getting Started
1. Clone the Repository
bash

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

Replace your-username and your-repo-name with your GitHub username and repository name.
2. Install Dependencies
bash

pip install opencv-python ultralytics

3. Prepare Your Video
Place your input video (e.g., your_video.mp4) in the inputs/ directory. Update the video_path in script.py to point to your video file.
4. Run the Script
Run the script to extract frames, detect people, and generate annotations:
bash

python script.py

Configuration
Modify these variables in script.py:
video_path: Path to your input video (e.g., "inputs/your_video.mp4").

output_dir: Directory for frames and annotations (e.g., "output/").

interval: Frame extraction interval (default: 30).

Example:
python

video_path = "inputs/your_video.mp4"
output_dir = "output"
main(video_path, output_dir)

5. Output
The script generates:
:framed_picture: Frames (.jpg) in output/.

:page_facing_up: YOLO annotations (.txt) for each frame.

:page_with_curl: Pascal VOC XML annotations (.xml) for each frame.

:mag: Manual Annotation with imgLabeler
To refine pre-generated annotations, use imgLabeler (labelImg).
How to Get imgLabeler
Clone the Repository:
bash

git clone https://github.com/tzutalin/labelImg.git
cd labelImg

Install:
bash

pip install -r requirements.txt
python labelImg.py

Use imgLabeler:
Open labelImg and select Open Dir to load the output/ directory.

Verify and edit bounding boxes for people.

Save changes to update XML files.

See imgLabeler GitHub for more details.
:open_file_folder: Project Structure

your-repo-name/
├── inputs/                # Input videos
│   └── your_video.mp4
├── output/                # Frames and annotations
│   ├── frame_0000.jpg
│   ├── frame_0000.txt
│   ├── frame_0000.xml
│   └── ...
├── script.py              # Main script
└── README.md              # This file

:bulb: Tips
:brain: Use yolov8m.pt or yolov8l.pt for higher accuracy (edit model = YOLO('yolov8n.pt') in script.py).

:timer_clock: Adjust interval in extract_frames() to control frame extraction frequency.

:mag_right: Always verify YOLO detections in imgLabeler for accuracy.

:bug: Troubleshooting
Video not found: Ensure video_path is correct.

Module not found: Install dependencies with pip install opencv-python ultralytics.

YOLO model issues: Ensure internet access for model download.

imgLabeler issues: Check imgLabeler documentation.

:handshake: Contributing
Contributions are welcome! Open issues or pull requests to improve the script or docs.
:scroll: License
This project is licensed under the MIT License. See the LICENSE file.
Happy labeling! :tada:

