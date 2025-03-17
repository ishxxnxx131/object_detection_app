// static/js/script.js
document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const fileInput = document.getElementById('file-input');
    const chooseFileBtn = document.getElementById('choose-file-btn');
    const fileNameSpan = document.getElementById('file-name');
    const uploadBtn = document.getElementById('upload-btn');
    const confidenceThreshold = document.getElementById('confidence-threshold');
    const thresholdValue = document.getElementById('threshold-value');
    const showBoxes = document.getElementById('show-boxes');
    const showLabels = document.getElementById('show-labels');
    const showConfidence = document.getElementById('show-confidence');
    const previewImage = document.getElementById('preview-image');
    const detectionCanvas = document.getElementById('detection-canvas');
    const detectionResults = document.getElementById('detection-results');
    const noResults = document.getElementById('no-results');
    const classFilters = document.getElementById('class-filters');
    
    // File selection
    chooseFileBtn.addEventListener('click', function() {
        fileInput.click();
    });
    
    fileInput.addEventListener('change', function() {
        if(this.files && this.files[0]) {
            const file = this.files[0];
            fileNameSpan.textContent = file.name;
            
            // Preview image
            const reader = new FileReader();
            reader.onload = function(e) {
                previewImage.src = e.target.result;
                previewImage.onload = function() {
                    // Clear previous detection results
                    clearDetectionCanvas();
                    detectionResults.innerHTML = '';
                    noResults.style.display = 'block';
                    
                    // Reset canvas dimensions to match image
                    detectionCanvas.width = previewImage.width;
                    detectionCanvas.height = previewImage.height;
                };
            };
            reader.readAsDataURL(file);
        }
    });
    
    // Update threshold value display
    confidenceThreshold.addEventListener('input', function() {
        thresholdValue.textContent = this.value;
    });
    
    // Toggle visibility of detection elements
    showBoxes.addEventListener('change', redrawDetections);
    showLabels.addEventListener('change', redrawDetections);
    showConfidence.addEventListener('change', redrawDetections);
    
    // Upload and detect
    uploadBtn.addEventListener('click', function() {
        if (!fileInput.files || !fileInput.files[0]) {
            alert('Please select a file first');
            return;
        }
        
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        formData.append('confidence_threshold', confidenceThreshold.value);
        
        // Show loading state
        uploadBtn.disabled = true;
        uploadBtn.textContent = 'Processing...';// Send request to server
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            // Reset button state
            uploadBtn.disabled = false;
            uploadBtn.textContent = 'Upload';
            
            if (data.error) {
                alert('Error: ' + data.error);
                return;
            }
            
            // Display results
            displayDetectionResults(data);
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred during processing');
            uploadBtn.disabled = false;
            uploadBtn.textContent = 'Upload';
        });
    });
    
    // Store current detections
    let currentDetections = [];
    let classColorMap = {};
    
    // Display detection results
    function displayDetectionResults(data) {
        currentDetections = data.results;
        
        // Clear previous results
        detectionResults.innerHTML = '';
        noResults.style.display = 'none';
        
        // Generate class color map if needed
        generateClassColors(data.results);
        
        // Display object count
        const totalObjects = data.results.length;
        detectionResults.innerHTML = `<p>Found ${totalObjects} objects:</p>`;
        
        // Create list of detected objects grouped by class
        const objectList = document.createElement('ul');
        
        Object.entries(data.class_counts).forEach(([className, count]) => {
            const listItem = document.createElement('li');
            const colorSwatch = document.createElement('span');
            colorSwatch.className = 'detection-color';
            colorSwatch.style.backgroundColor = classColorMap[className];
            
            listItem.appendChild(colorSwatch);
            listItem.appendChild(document.createTextNode(`${className}: ${count}`));
            objectList.appendChild(listItem);
            
            // Add class to filters if not already present
            addClassToFilters(className);
        });
        
        detectionResults.appendChild(objectList);
        
        // Draw detections on canvas
        drawDetections();
    }
    
    // Draw bounding boxes and labels on canvas
    function drawDetections() {
        clearDetectionCanvas();
        
        if (!showBoxes.checked && !showLabels.checked) {
            return;
        }
        
        const ctx = detectionCanvas.getContext('2d');
        const imageWidth = previewImage.width;
        const imageHeight = previewImage.height;
        
        // Adjust canvas size to match image
        detectionCanvas.width = imageWidth;
        detectionCanvas.height = imageHeight;
        
        currentDetections.forEach(detection => {
            const [x, y, w, h] = detection.box;
            const className = detection.class;
            const confidence = detection.confidence;
            
            // Check if class is filtered
            const classFilter = document.getElementById(`filter-${className}`);
            if (classFilter && !classFilter.checked) {
                return;
            }
            
            const color = classColorMap[className];
            
            // Draw bounding box
            if (showBoxes.checked) {
                ctx.strokeStyle = color;
                ctx.lineWidth = 2;
                ctx.strokeRect(x, y, w, h);
            }
            
            // Draw label
            if (showLabels.checked) {
                const label = showConfidence.checked 
                    ? `${className} (${confidence})` 
                    : className;
                
                ctx.fillStyle = color;
                const textWidth = ctx.measureText(label).width;
                
                ctx.fillRect(x, y - 20, textWidth + 10, 20);
                
                ctx.font = '16px Arial';
                ctx.fillStyle = 'white';
                ctx.fillText(label, x + 5, y - 5);
            }
        });
    }
    
    // Clear detection canvas
    function clearDetectionCanvas() {
        const ctx = detectionCanvas.getContext('2d');
        ctx.clearRect(0, 0, detectionCanvas.width, detectionCanvas.height);
    }
    
    // Redraw detections when visibility toggles change
    function redrawDetections() {
        if (currentDetections.length > 0) {
            drawDetections();
        }
    }
    
    // Generate consistent colors for classes
    function generateClassColors(detections) {
        // Get unique classes
        const classes = [...new Set(detections.map(d => d.class))];
        
        // Generate colors for new classes
        classes.forEach(className => {
            if (!classColorMap[className]) {
                const hue = Math.random() * 360;
                classColorMap[className] = `hsl(${hue}, 70%, 50%)`;
            }
        });
    }
    
    // Add class to filters panel
    function addClassToFilters(className) {
        // Check if filter already exists
        if (document.getElementById(`filter-${className}`)) {
            return;
        }
        
        const filterContainer = document.createElement('div');
        filterContainer.className = 'checkbox-container';
        
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.id = `filter-${className}`;
        checkbox.checked = true;
        checkbox.addEventListener('change', redrawDetections);
        
        const label = document.createElement('label');
        label.htmlFor = `filter-${className}`;
        
        const colorSwatch = document.createElement('span');
        colorSwatch.className = 'detection-color';
        colorSwatch.style.backgroundColor = classColorMap[className];
        
        label.appendChild(colorSwatch);
        label.appendChild(document.createTextNode(` ${className}`));
        
        filterContainer.appendChild(checkbox);
        filterContainer.appendChild(label);
        
        classFilters.appendChild(filterContainer);
    }
});