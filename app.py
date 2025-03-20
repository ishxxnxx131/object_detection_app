# app.py
from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from flask import Response
import threading
import time


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

camera = None
output_frame = None
lock = threading.Lock()

def get_yolo_classes():
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return classes

# Load YOLO model
def load_yolo():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    
    return net, classes, output_layers

# Detect objects in image
def detect_objects(img_path, confidence_threshold=0.5):
    net, classes, output_layers = load_yolo()
    
    # Load image
    img = cv2.imread(img_path)
    height, width, channels = img.shape
    
    # Preprocess image for YOLO
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
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
    
    # Count occurrences of each class
    class_counts = {}
    for result in results:
        class_name = result['class']
        if class_name in class_counts:
            class_counts[class_name] += 1
        else:
            class_counts[class_name] = 1
    
    return results, class_counts

@app.route('/')
def index():
    classes = get_yolo_classes()
    return render_template('index.html', classes=classes)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Get confidence threshold
        confidence_threshold = float(request.form.get('confidence_threshold', 0.5))
        
        # Process the image
        results, class_counts = detect_objects(file_path, confidence_threshold)
        
        return jsonify({
            'filename': filename,
            'results': results,
            'class_counts': class_counts
        })

# Add this function to handle webcam video stream
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

# function to detect objects in webcam stream
def detect_webcam(confidence_threshold=0.5, class_filters=None):
    global camera, output_frame, lock
    
    net, classes, output_layers = load_yolo()
    
    # If no class filters provided, use all classes
    if class_filters is None:
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
                    # Draw rectangle and text
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Update the output frame
        with lock:
            output_frame = frame.copy()

# Add these routes to app.py
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
    
    # Get confidence threshold
    confidence_threshold = float(request.form.get('confidence_threshold', 0.5))
    
    # Get class filters (if provided)
    class_filters = request.form.getlist('class_filters[]')
    if not class_filters:  # If no filters, use all classes
        class_filters = all_classes
    
    # Start webcam detection in a separate thread
    t = threading.Thread(target=detect_webcam, args=(confidence_threshold, class_filters))
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


# global variable 
all_classes = []

# Add this function to load all classes at startup
def load_all_classes():
    global all_classes
    try:
        with open("coco.names", "r") as f:
            all_classes = [line.strip() for line in f.readlines()]
        return all_classes
    except Exception as e:
        print(f"Error loading classes: {e}")
        return []

# Add this route to get all classes
@app.route('/get_classes', methods=['GET'])
def get_classes():
    return jsonify({'classes': all_classes})

# In your main block, add this before app.run()
if __name__ == '__main__':
    # Load all classes at startup
    all_classes = load_all_classes()
    app.run(debug=True)


    