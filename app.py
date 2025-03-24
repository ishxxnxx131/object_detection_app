from flask import Flask, render_template, request, jsonify, Response
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import threading
import time
import json
import colorsys

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max upload
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for webcam
camera = None
output_frame = None
lock = threading.Lock()

# Global variable for classes
all_classes = []

# Load all classes at startup
def load_all_classes():
    global all_classes
    try:
        with open("coco.names", "r") as f:
            all_classes = [line.strip() for line in f.readlines()]
        return all_classes
    except Exception as e:
        print(f"Error loading classes: {e}")
        return []

# Load YOLO model
def load_yolo(model="YOLOv3"):
    if model == "YOLOv3":
        weights = "yolov3.weights"
        config = "yolov3.cfg"
    elif model == "YOLOv4":
        weights = "yolov4.weights"
        config = "yolov4.cfg"
    elif model == "YOLOv8":
        weights = "yolov8.weights"
        config = "yolov8.cfg"
    else:
        weights = "yolov3.weights"
        config = "yolov3.cfg"
    
    net = cv2.dnn.readNet(weights, config)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    
    return net, all_classes, output_layers

# Process a single frame
def process_frame(frame, net, classes, output_layers, confidence_threshold=0.5):
    height, width, channels = frame.shape
    
    # Preprocess image for YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    # Process detections
    class_ids = []
    confidences = []
    boxes = []
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > confidence_threshold:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Apply non-max suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.4)
    
    results = []
    if len(indexes) > 0:
        for i in indexes.flatten():
            results.append({
                'class': classes[class_ids[i]],
                'confidence': round(confidences[i], 2),
                'box': boxes[i]
            })
    
    return results

# Check if file is a video
def is_video_file(file_path):
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
    ext = os.path.splitext(file_path)[1].lower()
    return ext in video_extensions

# Detect objects in image or video
def detect_objects(file_path, confidence_threshold=0.5, model="YOLOv3"):
    net, classes, output_layers = load_yolo(model)
    
    # Check if file is a video
    if is_video_file(file_path):
        return process_video(file_path, net, classes, output_layers, confidence_threshold)
    else:
        return process_image(file_path, net, classes, output_layers, confidence_threshold)

# Process image file
def process_image(img_path, net, classes, output_layers, confidence_threshold=0.5):
    # Load image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not load image from {img_path}")
    
    # Process the frame
    results = process_frame(img, net, classes, output_layers, confidence_threshold)
    
    # Count occurrences of each class
    class_counts = {}
    for result in results:
        class_name = result['class']
        if class_name in class_counts:
            class_counts[class_name] += 1
        else:
            class_counts[class_name] = 1
    
    return results, class_counts

# Process video file
def process_video(video_path, net, classes, output_layers, confidence_threshold=0.5):
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    all_results = []
    frame_count = 0
    sample_interval = 10  # Process every 10th frame
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Only process every sample_interval frames
        if frame_count % sample_interval == 0:
            # Process the frame
            results = process_frame(frame, net, classes, output_layers, confidence_threshold)
            all_results.extend(results)
        
        frame_count += 1
    
    # Release video capture
    cap.release()
    
    # Count occurrences of each class across all processed frames
    class_counts = {}
    for result in all_results:
        class_name = result['class']
        if class_name in class_counts:
            class_counts[class_name] += 1
        else:
            class_counts[class_name] = 1
    
    return all_results, class_counts

# Function to detect objects in webcam stream
def detect_webcam(confidence_threshold=0.5, class_filters=None, model="YOLOv3"):
    global camera, output_frame, lock
    
    net, classes, output_layers = load_yolo(model)
    
    # If no class filters provided, use all classes
    if class_filters is None or len(class_filters) == 0:
        class_filters = classes
    
    # Initialize webcam
    camera = cv2.VideoCapture(0)
    time.sleep(2.0)  # Allow camera to warm up
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Preprocess image for YOLO
        height, width, channels = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)
        
        # Process detections
        class_ids = []
        confidences = []
        boxes = []
        
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > confidence_threshold:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply non-max suppression
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.4)
        
        # Draw bounding boxes and labels
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                
                # Draw only if class is in filters
                if label in class_filters:
                    # Generate a consistent color based on class
                    color = get_color_for_class(label)
                    
                    # Draw rectangle and text
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Update the output frame
        with lock:
            output_frame = frame.copy()

# Function to generate video frames for streaming
def generate_frames():
    global output_frame, camera
    
    while True:
        with lock:
            if output_frame is None:
                continue
            
            # Encode the frame as JPEG
            (flag, encoded_image) = cv2.imencode(".jpg", output_frame)
            if not flag:
                continue
            
        # Yield the output frame in the byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
              bytearray(encoded_image) + b'\r\n')

# Function to generate consistent colors for classes
def get_color_for_class(class_name):
    # Simple hash function to generate consistent colors
    hash_val = sum(ord(c) for c in class_name)
    hue = hash_val % 360
    # Convert HSV to BGR (what OpenCV uses)
    h = hue / 360.0
    rgb = tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h, 0.7, 0.9))
    # OpenCV uses BGR
    return (rgb[2], rgb[1], rgb[0])

# Routes

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        try:
            # Save the uploaded file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Get confidence threshold and model
            confidence_threshold = float(request.form.get('confidence_threshold', 0.5))
            model = request.form.get('model', 'YOLOv3')
            
            # Process the file (image or video)
            results, class_counts = detect_objects(file_path, confidence_threshold, model)
            
            return jsonify({
                'filename': filename,
                'results': results,
                'class_counts': class_counts,
                'is_video': is_video_file(file_path)
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

@app.route('/get_classes', methods=['GET'])
def get_classes():
    return jsonify({'classes': all_classes})

@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_webcam', methods=['POST'])
def start_webcam():
    global camera
    
    # If camera is already running, stop it
    if camera is not None:
        camera.release()
        camera = None
    
    # Get parameters from request
    data = request.get_json() if request.is_json else request.form
    
    # Get confidence threshold
    confidence_threshold = float(data.get('confidence_threshold', 0.5))
    
    # Get model selection
    model = data.get('model', 'YOLOv3')
    
    # Get class filters (if provided)
    class_filters = data.getlist('class_filters[]') if hasattr(data, 'getlist') else data.get('class_filters', [])
    
    # Start webcam detection in a separate thread
    t = threading.Thread(target=detect_webcam, args=(confidence_threshold, class_filters, model))
    t.daemon = True
    t.start()
    
    return jsonify({'status': 'success'})

@app.route('/stop_webcam', methods=['POST'])
def stop_webcam():
    global camera
    
    # Stop the webcam
    if camera is not None:
        camera.release()
        camera = None

    return jsonify({'status': 'success'})

@app.route('/local_repository', methods=['GET'])
def local_repository():
    # Get list of files in the local repository (uploads folder)
    files = []
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        if os.path.isfile(os.path.join(app.config['UPLOAD_FOLDER'], filename)):
            # Check if it's an image or video
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.mp4', '.avi', '.mov')):
                files.append({
                    'name': filename,
                    'path': os.path.join(app.config['UPLOAD_FOLDER'], filename),
                    'is_video': is_video_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                })
    
    return jsonify({'files': files})

@app.route('/analyze_file', methods=['POST'])
def analyze_file():
    data = request.get_json()
    
    # Get file path, confidence threshold, and model
    file_path = data.get('file_path')
    confidence_threshold = float(data.get('confidence_threshold', 0.5))
    model = data.get('model', 'YOLOv3')
    
    if not file_path or not os.path.exists(file_path):
        return jsonify({'error': 'File not found'})
    
    try:
        # Process the file
        results, class_counts = detect_objects(file_path, confidence_threshold, model)
        
        return jsonify({
            'filename': os.path.basename(file_path),
            'results': results,
            'class_counts': class_counts,
            'is_video': is_video_file(file_path)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Handle cleanup when the application shuts down
@app.teardown_appcontext
def teardown_app(exception=None):
    global camera
    if camera is not None:
        camera.release()

if __name__ == '__main__':
    # Load all classes at startup
    all_classes = load_all_classes()
    app.run(debug=True)
