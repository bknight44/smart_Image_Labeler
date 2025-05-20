import cv2
import os
from ultralytics import YOLO
import glob
import xml.etree.ElementTree as ET

def extract_frames(video_path, output_dir, interval=30):
    """Extract frames from a video at a specified interval."""
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{count:04d}.jpg")
            cv2.imwrite(frame_path, frame)
        count += 1
    cap.release()
    print(f"Extracted frames to {output_dir}")

def yolo_to_voc(txt_path, img_path, output_xml_path):
    """Convert YOLO format detections to Pascal VOC XML format."""
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    
    img = cv2.imread(img_path)
    height, width, _ = img.shape
    
    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'filename').text = os.path.basename(img_path)
    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(width)
    ET.SubElement(size, 'height').text = str(height)
    ET.SubElement(size, 'depth').text = '3'
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 5:  # Ensure valid YOLO format line
            class_id, x_center, y_center, w, h = map(float, parts)
            x_min = int((x_center - w / 2) * width)
            y_min = int((y_center - h / 2) * height)
            x_max = int((x_center + w / 2) * width)
            y_max = int((y_center + h / 2) * height)
            
            object_elem = ET.SubElement(annotation, 'object')
            ET.SubElement(object_elem, 'name').text = 'person'
            ET.SubElement(object_elem, 'pose').text = 'Unspecified'
            ET.SubElement(object_elem, 'truncated').text = '0'
            ET.SubElement(object_elem, 'difficult').text = '0'
            bndbox = ET.SubElement(object_elem, 'bndbox')
            ET.SubElement(bndbox, 'xmin').text = str(x_min)
            ET.SubElement(bndbox, 'ymin').text = str(y_min)
            ET.SubElement(bndbox, 'xmax').text = str(x_max)
            ET.SubElement(bndbox, 'ymax').text = str(y_max)
    
    tree = ET.ElementTree(annotation)
    tree.write(output_xml_path)

def main(video_path, output_dir):
    # Step 1: Extract frames
    extract_frames(video_path, output_dir)
    
    # Step 2: Load YOLOv8 model and detect people
    model = YOLO('yolov8n.pt')  # Using the nano version for speed; use 'yolov8m.pt' or 'yolov8l.pt' for higher accuracy
    
    frame_paths = glob.glob(os.path.join(output_dir, "*.jpg"))
    for frame_path in frame_paths:
        # Run YOLOv8 detection
        results = model(frame_path)
        
        # Save detections in YOLO format
        txt_path = frame_path.replace('.jpg', '.txt')
        with open(txt_path, 'w') as f:
            for det in results[0].boxes:
                if det.cls.item() == 0:  # Class 0 is 'person' in COCO dataset
                    x_center, y_center, width, height = det.xywh[0]
                    img_width, img_height = results[0].orig_shape[1], results[0].orig_shape[0]
                    x_center /= img_width
                    y_center /= img_height
                    width /= img_width
                    height /= img_height
                    f.write(f"0 {x_center} {y_center} {width} {height}\n")
        
        # Step 3: Convert to Pascal VOC XML
        xml_path = frame_path.replace('.jpg', '.xml')
        yolo_to_voc(txt_path, frame_path, xml_path)
    
    # Step 4: Instructions for manual verification
    print(f"\nPre-labeling complete! Frames and annotations are saved in {output_dir}.")
    print("Next steps:")
    print("1. Open labelImg by running 'labelImg' in your terminal.")
    print(f"2. Select 'Open Dir' in labelImg and choose the directory: {output_dir}")
    print("3. Verify and correct the pre-detected bounding boxes as needed.")
    print("4. Save your changes in labelImg (annotations will be updated in the XML files).")

if __name__ == "__main__":
    # Replace these paths with your own
    video_path = "inputs/output_video_all_3.mp4"  # Path to your input video
    output_dir = "output"  # Directory where frames and annotations will be saved
    main(video_path, output_dir)