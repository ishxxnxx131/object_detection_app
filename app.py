import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from werkzeug.utils import secure_filename
import threading
import time
import colorsys

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max upload
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for webcam and video processing
camera = None
output_frame = None
lock = threading.Lock()
current_video = None
current_video_frame = None
video_frame_count = 0
video_fps = 30
is_video_playing = False
current_video_path = None

# Global variable for classes
all_classes = []

def load_all_classes():
    """Load classes from COCO names file."""
    global all_classes
    try:
        with open("coco.names", "r") as f:
            all_classes = [line.strip() for line in f.readlines()]
        return all_classes
    except Exception as e:
        print(f"Error loading classes: {e}")
        return []

def load_yolo(model="YOLOv3"):
    """Load YOLO model based on selected version."""
    model_configs = {
        "YOLOv3": ("yolov3.weights", "yolov3.cfg"),
        "YOLOv4": ("yolov4.weights", "yolov4.cfg"),
        "YOLOv8": ("yolov8.weights", "yolov8.cfg")
    }
    
    weights, config = model_configs.get(model, model_configs["YOLOv3"])
    
    net = cv2.dnn.readNet(weights, config)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    
    return net, all_classes, output_layers

class ObjectTracker:
    def __init__(self, max_lost_frames=30):
        self.tracks = []
        self.next_id = 0
        self.max_lost_frames = max_lost_frames

    def update(self, detections):
        # Update existing tracks
        for track in self.tracks[:]:
            track['age'] += 1
            track['matched'] = False

        # Match new detections to existing tracks
        for detection in detections:
            best_match = None
            best_iou = 0.3  # Minimum IoU to consider a match

            for track in self.tracks:
                if track['class'] == detection['class']:
                    iou = calculate_iou(detection['box'], track['box'])
                    if iou > best_iou:
                        best_match = track
                        best_iou = iou

            if best_match:
                # Update matched track
                best_match['box'] = detection['box']
                best_match['confidence'] = detection['confidence']
                best_match['age'] = 0
                best_match['matched'] = True
            else:
                # Create new track
                self.tracks.append({
                    'id': self.next_id,
                    'class': detection['class'],
                    'box': detection['box'],
                    'confidence': detection['confidence'],
                    'age': 0,
                    'matched': True
                })
                self.next_id += 1

        # Remove lost tracks
        self.tracks = [
            track for track in self.tracks 
            if track['matched'] or track['age'] < self.max_lost_frames
        ]

        # Prepare results with track IDs
        results = []
        for track in self.tracks:
            if track['matched']:
                results.append({
                    'class': track['class'],
                    'confidence': track['confidence'],
                    'box': track['box'],
                    'track_id': track['id']
                })

        return results

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Convert to coordinate format
    b1 = [x1, y1, x1+w1, y1+h1]
    b2 = [x2, y2, x2+w2, y2+h2]
    
    # Calculate intersection coordinates
    x_left = max(b1[0], b2[0])
    y_top = max(b1[1], b2[1])
    x_right = min(b1[2], b2[2])
    y_bottom = min(b1[3], b2[3])
    
    # Calculate intersection area
    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)
    
    # Calculate union area
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0
    return iou

def get_color_for_class(class_name):
    """Generate consistent color for each class."""
    hash_val = sum(ord(c) for c in class_name)
    hue = hash_val % 360
    h = hue / 360.0
    rgb = tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h, 0.7, 0.9))
    return (rgb[2], rgb[1], rgb[0])

def draw_detections(frame, detections):
    """Draw bounding boxes and labels on a frame."""
    for detection in detections:
        x, y, w, h = detection['box']
        label = f"{detection['class']} {detection['confidence']:.2f} (ID:{detection.get('track_id', 'N/A')})"
        color = get_color_for_class(detection['class'])
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return frame

def process_frame_with_tracking(frame, net, classes, output_layers, tracker, confidence_threshold=0.5):
    """Process a single frame with object detection and tracking."""
    height, width, channels = frame.shape
    
    # Perform object detection
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    class_ids, confidences, boxes = [], [], []
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > confidence_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                # Filter for specific classes
                if classes[class_id] in ['car', 'truck', 'bus', 'motorcycle', 'person']:
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
    
    # Non-maximum suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.4)
    
    # Prepare detections
    detections = []
    if len(indexes) > 0:
        for i in indexes.flatten():
            detections.append({
                'class': classes[class_ids[i]],
                'confidence': round(confidences[i], 2),
                'box': boxes[i]
            })
    
    # Update tracker and get tracked results
    tracked_results = tracker.update(detections)
    
    return tracked_results

def process_frame(frame, net, classes, output_layers, confidence_threshold=0.5):
    """Process a single frame for object detection."""
    height, width, channels = frame.shape
    
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    class_ids, confidences, boxes = [], [], []
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > confidence_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
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

def process_video_detection(video_path, net, classes, output_layers, confidence_threshold=0.5):
    """Process video for object detection with frame-by-frame analysis."""
    cap = cv2.VideoCapture(video_path)
    all_results = {}
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for frame_num in range(0, total_frames, max(1, total_frames // 20)):  # Sample 20 frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            break
        
        results = process_frame(frame, net, classes, output_layers, confidence_threshold)
        all_results[frame_num] = results
    
    cap.release()
    return all_results

def video_stream_generator(video_path):
    """Generate video frames with object detection and tracking."""
    global current_video_frame, video_frame_count, is_video_playing
    
    net, classes, output_layers = load_yolo()
    cap = cv2.VideoCapture(video_path)
    
    # Initialize object tracker
    tracker = ObjectTracker()
    
    while True:
        if not is_video_playing:
            time.sleep(0.1)
            continue
        
        ret, frame = cap.read()
        if not ret:
            break
        
        # Perform object detection with tracking
        detections = process_frame_with_tracking(
            frame, net, classes, output_layers, tracker
        )
        
        # Draw detections on frame
        frame_with_detections = draw_detections(frame, detections)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame_with_detections)
        frame_bytes = buffer.tobytes()
        
        current_video_frame = frame_bytes
        video_frame_count += 1
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()

# Routes for the application
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global current_video_path
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    net, classes, output_layers = load_yolo()
    
    try:
        if filename.lower().endswith(('mp4', 'avi', 'mov', 'mkv')):
            # Store current video path for streaming
            current_video_path = filepath
            
            # For video, sample detection across frames
            results = process_video_detection(filepath, net, classes, output_layers)
            
            # Prepare results in a format compatible with frontend
            formatted_results = []
            class_counts = {}
            for frame_num, frame_results in results.items():
                for result in frame_results:
                    formatted_results.append({
                        'frame': frame_num,
                        **result
                    })
                    
                    # Count classes
                    class_name = result['class']
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            return jsonify({
                'filename': filename,
                'results': formatted_results,
                'class_counts': class_counts,
                'is_video': True
            })
        else:
            # For images
            results = process_frame(cv2.imread(filepath), net, classes, output_layers)
            
            # Count classes
            class_counts = {}
            for result in results:
                class_name = result['class']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            return jsonify({
                'filename': filename,
                'results': results,
                'class_counts': class_counts,
                'is_video': False
            })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/video_stream')
def video_stream():
    """Stream video with object detection."""
    global current_video_path
    if not current_video_path:
        return "No video uploaded", 400
    
    return Response(video_stream_generator(current_video_path), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_video_play', methods=['POST'])
def toggle_video_play():
    """Toggle video play/pause."""
    global is_video_playing
    data = request.get_json()
    is_video_playing = data.get('playing', False)
    return jsonify({'status': 'success', 'playing': is_video_playing})

@app.route('/get_classes', methods=['GET'])
def get_classes():
    """Return list of available classes."""
    return jsonify({'classes': all_classes})

if __name__ == '__main__':
    # Load classes at startup
    all_classes = load_all_classes()
    app.run(debug=True)
