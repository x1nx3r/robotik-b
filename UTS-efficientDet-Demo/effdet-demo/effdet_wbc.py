import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import os
import time
from collections import deque

# Define model cache directory in the project folder
MODEL_URL = "https://tfhub.dev/tensorflow/efficientdet/d0/1"
# Use the project directory to save the model
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_CACHE_DIR = os.path.join(PROJECT_DIR, "models")
MODEL_LOCAL_PATH = os.path.join(MODEL_CACHE_DIR, "efficientdet_d0")

# Load the EfficientDet-D0 model (check local cache first)
print("Loading model...")
if os.path.exists(MODEL_LOCAL_PATH):
    print(f"Loading model from local cache: {MODEL_LOCAL_PATH}")
    model = tf.saved_model.load(MODEL_LOCAL_PATH)
else:
    print(f"Downloading model from TF Hub: {MODEL_URL}")
    model = hub.load(MODEL_URL)
    # Save the model locally
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    tf.saved_model.save(model, MODEL_LOCAL_PATH)
    print(f"Model saved to: {MODEL_LOCAL_PATH}")
print("Model loaded!")

# Load label map (COCO labels)
COCO_LABELS = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

def draw_boxes(frame, boxes, classes, scores, threshold=0.3):
    height, width, _ = frame.shape
    for box, cls, score in zip(boxes, classes, scores):
        if score < threshold:
            continue
        ymin, xmin, ymax, xmax = box
        (left, right, top, bottom) = (int(xmin * width), int(xmax * width),
                                      int(ymin * height), int(ymax * height))
        
        # Fix the class index issue
        class_idx = int(cls)
        if 1 <= class_idx <= len(COCO_LABELS):
            label = f"{COCO_LABELS[class_idx-1]}: {score:.2f}"
        else:
            label = f"Class {class_idx}: {score:.2f}"
            
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, label, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

# Open webcam
cap = cv2.VideoCapture(0)

# Add a rectangle to show inference area
show_inference_area = True

# Variables for FPS calculation
prev_frame_time = 0
curr_frame_time = 0
fps = 0
# Use a deque for rolling average FPS calculation
fps_buffer = deque(maxlen=30)  # Store last 30 frames for smoother FPS display

# Frame skipping variables
frame_skip_enabled = True
frame_count = 0
skip_frames = 0  # Dynamically adjusted based on FPS
target_fps = 15.0  # Target FPS we want to maintain
last_annotated_frame = None

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Calculate FPS
    curr_frame_time = time.time()
    time_diff = curr_frame_time - prev_frame_time
    if time_diff > 0:
        curr_fps = 1.0 / time_diff
        fps_buffer.append(curr_fps)
        fps = sum(fps_buffer) / len(fps_buffer)  # Average FPS for smoother display
    prev_frame_time = curr_frame_time
    
    # Dynamically adjust frame skipping based on current FPS
    if frame_skip_enabled and fps > 0:
        if fps < target_fps / 2:
            skip_frames = min(skip_frames + 1, 4)  # Max skip 4 frames
        elif fps > target_fps * 1.2:
            skip_frames = max(skip_frames - 1, 0)  # Min skip 0 frames
    else:
        skip_frames = 0
    
    frame_count += 1
    
    # Skip frames if needed to maintain performance
    if frame_skip_enabled and skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
        # Use the last annotated frame but update the FPS display
        if last_annotated_frame is not None:
            # Update FPS display on the last processed frame
            display_frame = last_annotated_frame.copy()
            cv2.putText(display_frame, f"FPS: {fps:.1f} (skipping {skip_frames} frames)", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("EfficientDet-D0 Webcam (CPU)", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            elif cv2.waitKey(1) & 0xFF == ord('s'):
                # Toggle frame skipping with 's' key
                frame_skip_enabled = not frame_skip_enabled
            continue
    
    # Get original frame dimensions
    height, width, _ = frame.shape
    
    # Calculate center crop coordinates
    start_y = max(0, (height - 512) // 2)
    start_x = max(0, (width - 512) // 2)
    end_y = min(height, start_y + 512)
    end_x = min(width, start_x + 512)
    
    # Extract center crop for inference
    center_crop = frame[start_y:end_y, start_x:end_x].copy()
    
    # Ensure the crop is exactly 512x512 (in case of smaller frames)
    if center_crop.shape[0] != 512 or center_crop.shape[1] != 512:
        center_crop = cv2.resize(center_crop, (512, 512))
    
    # Preprocess center crop for inference
    img_normalized = tf.cast(center_crop, tf.uint8)
    input_tensor = tf.expand_dims(img_normalized, axis=0)

    # Run inference
    outputs = model(input_tensor)

    # Extract results
    boxes = outputs["detection_boxes"].numpy()[0]
    class_ids = outputs["detection_classes"].numpy()[0].astype(np.int32)
    scores = outputs["detection_scores"].numpy()[0]

    # Create a copy of the frame for drawing
    annotated_frame = frame.copy()
    
    # Draw boxes (adjust coordinates to the center crop region)
    crop_height, crop_width, _ = center_crop.shape
    for box, cls, score in zip(boxes, class_ids, scores):
        if score < 0.3:  # Using the threshold
            continue
            
        ymin, xmin, ymax, xmax = box
        
        # Map normalized coordinates to the crop region in the full frame
        left = int(xmin * crop_width) + start_x
        right = int(xmax * crop_width) + start_x
        top = int(ymin * crop_height) + start_y
        bottom = int(ymax * crop_height) + start_y
        
        # Fix the class index issue
        class_idx = int(cls)
        if 1 <= class_idx <= len(COCO_LABELS):
            label = f"{COCO_LABELS[class_idx-1]}: {score:.2f}"
        else:
            label = f"Class {class_idx}: {score:.2f}"
            
        cv2.rectangle(annotated_frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(annotated_frame, label, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Draw inference area rectangle
    if show_inference_area:
        cv2.rectangle(annotated_frame, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)
        cv2.putText(annotated_frame, "Inference Area", (start_x, start_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Draw FPS and frame skipping info
    skipping_text = f" (skipping {skip_frames} frames)" if frame_skip_enabled and skip_frames > 0 else ""
    cv2.putText(annotated_frame, f"FPS: {fps:.1f}{skipping_text}", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Save the current frame for displaying during skipped frames
    last_annotated_frame = annotated_frame.copy()
    
    # Display
    cv2.imshow("EfficientDet-D0 Webcam (CPU)", annotated_frame)
    
    # Check for key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        # Toggle frame skipping with 's' key
        frame_skip_enabled = not frame_skip_enabled
        print(f"Frame skipping {'enabled' if frame_skip_enabled else 'disabled'}")

cap.release()
cv2.destroyAllWindows()