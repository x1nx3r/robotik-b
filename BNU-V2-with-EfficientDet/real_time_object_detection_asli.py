# Excerpt from:
# Excerpt from:
# To run: python3 real_time_object_detection_asli.py

import time
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import serial

# Hardcoded model path and parameters
MODEL_PATH = "/media/hdd2/git/robotika/Robot_Surveillance/efficientdet_lite0.tflite"  # Full path
PROBABILITY_THRESHOLD = 0.2

# Arduino setup
print("[INFO] connecting to Arduino...")
arduino = serial.Serial('/dev/ttyUSB0', 115200)# Adjust COM port as needed (linux uses /dev/... so check that)
time.sleep(2)

# Complete COCO class labels (91 classes, 0 is background)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table',
    '', '', 'toilet', '', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

# Load TFLite model and allocate tensors
print("[INFO] loading TFLite model...")
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

print(f"[DEBUG] Model input shape: {height}x{width}")
print(f"[DEBUG] COCO_CLASSES length: {len(COCO_CLASSES)}")

# Generate colors for each class
np.random.seed(42)
COLORS = np.random.uniform(0, 255, size=(len(COCO_CLASSES), 3))

print("[INFO] starting video stream...")
vs = cv2.VideoCapture(0)
time.sleep(2.0)

while True:
    ret, frame = vs.read()
    if not ret:
        break
    orig = frame.copy()
    img = cv2.resize(frame, (width, height))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_data = np.expand_dims(img_rgb, axis=0).astype(np.uint8)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Output details: boxes, class_ids, scores, num_detections
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # [N, 4]
    class_ids = interpreter.get_tensor(output_details[1]['index'])[0].astype(int)  # [N]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # [N]
    num = int(interpreter.get_tensor(output_details[3]['index'])[0])

    # Arduino control logic
    detected_objects = []

    h, w, _ = orig.shape
    for i in range(num):
        score = scores[i]
        if score < PROBABILITY_THRESHOLD:
            continue
        ymin, xmin, ymax, xmax = boxes[i]
        (startX, startY, endX, endY) = (int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h))
        class_idx = class_ids[i]

        # Debug information
        print(f"[DEBUG] Detection {i}: class_idx={class_idx}, score={score:.2f}, max_class_idx={len(COCO_CLASSES)-1}")

        # Bounds checking before accessing COCO_CLASSES
        if class_idx < len(COCO_CLASSES) and class_idx >= 0:
            class_name = COCO_CLASSES[class_idx]

            # Skip empty class names
            if class_name == '':
                print(f"[DEBUG] Skipping empty class at index {class_idx}")
                continue

            # Store detected object for Arduino control
            detected_objects.append(class_name)

            color = COLORS[class_idx % len(COLORS)]
            label_text = f"{class_name}: {score:.2f}"
        else:
            # Handle unknown/out-of-bounds class
            print(f"[WARNING] Class index {class_idx} is out of bounds (max: {len(COCO_CLASSES)-1})")
            class_name = f"unknown_class_{class_idx}"
            detected_objects.append(class_name)
            color = (255, 255, 255)  # White color for unknown
            label_text = f"{class_name}: {score:.2f}"

        # Draw bounding box and label
        cv2.rectangle(orig, (startX, startY), (endX, endY), color, 2)
        y = startY - 10 if startY - 10 > 10 else startY + 20
        cv2.putText(orig, label_text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Arduino control based on detected objects
    target_objects = [obj for obj in detected_objects if obj in ['bottle', 'person', 'chair']]

    if target_objects:
        print(f"[DEBUG] Sending STOP command (0) - Detected target objects: {target_objects}")
        arduino.write(b'0')  # Stop when detecting bottle, person, or chair
    else:
        print("[DEBUG] Sending ROTATE LEFT command (3) - No target objects detected, continuing search")
        arduino.write(b'3')  # Rotate left (continue searching)

    # Display all detected objects for debugging
    if detected_objects:
        print(f"[DEBUG] All detected objects: {detected_objects}")

    cv2.imshow("Frame", orig)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

vs.release()
cv2.destroyAllWindows()
arduino.close()
