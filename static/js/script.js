// static/js/script.js
document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const sourceSelect = document.getElementById('source-select');
    const fileUploadContainer = document.getElementById('file-upload-container');
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
    const previewVideo = document.getElementById('preview-video');
    const detectionCanvas = document.getElementById('detection-canvas');
    const analysisLogs = document.getElementById('analysis-logs');
    const classFilters = document.getElementById('class-filters');
    const modelSelect = document.getElementById('model-select');
    const loadingIndicator = document.getElementById('loading-indicator');
    const frameCounter = document.getElementById('frame-counter');
    const currentFile = document.querySelector('.current-file span');
    
    // Navigation buttons
    const firstFrameBtn = document.getElementById('first-frame');
    const prevFrameBtn = document.getElementById('prev-frame');
    const nextFrameBtn = document.getElementById('next-frame');
    const lastFrameBtn = document.getElementById('last-frame');
    const playBtn = document.getElementById('play-btn');
    
    // State variables
    let currentDetections = [];
    let classColorMap = {};
    let isPlaying = false;
    let currentFrame = 0;
    let totalFrames = 0;
    let activeSource = null;
    
    // Load all YOLO classes
    loadAllClasses();
    
    // Source selection event
    sourceSelect.addEventListener('change', function() {
        const selectedSource = this.value;
        
        // Reset UI state
        resetUIState();
        
        if (selectedSource === 'upload') {
            fileUploadContainer.classList.remove('hidden');
            activeSource = 'upload';
        } else if (selectedSource === 'webcam') {
            // Redirect to webcam page or initialize webcam
            initializeWebcam();
            activeSource = 'webcam';
        } else if (selectedSource === 'local') {
            // Handle local repository
            // This would typically require server-side implementation
            fileUploadContainer.classList.remove('hidden');
            activeSource = 'local';
        }
    });
    
    // File selection
    chooseFileBtn.addEventListener('click', function() {
        fileInput.click();
    });
    
    fileInput.addEventListener('change', function() {
        if(this.files && this.files[0]) {
            const file = this.files[0];
            fileNameSpan.textContent = file.name;
            currentFile.textContent = `Current file: ${file.name}`;
            
            // Check if file is video or image
            if (file.type.startsWith('video/')) {
                previewVideo.classList.remove('hidden');
                previewImage.classList.add('hidden');
                
                // Create object URL for video
                const videoURL = URL.createObjectURL(file);
                previewVideo.src = videoURL;
                
                // Get video metadata to set frame count
                previewVideo.onloadedmetadata = function() {
                    totalFrames = Math.floor(previewVideo.duration * 30); // Assuming 30fps
                    frameCounter.textContent = `1 / ${totalFrames}`;
                };
            } else {
                // Handle as image
                previewImage.classList.remove('hidden');
                previewVideo.classList.add('hidden');
                
                const reader = new FileReader();
                reader.onload = function(e) {
                    previewImage.src = e.target.result;
                    totalFrames = 1;
                    frameCounter.textContent = `1 / 1`;
                    
                    // Reset canvas dimensions to match image
                    previewImage.onload = function() {
                        clearDetectionCanvas();
                        detectionCanvas.width = previewImage.width;
                        detectionCanvas.height = previewImage.height;
                    };
                };
                reader.readAsDataURL(file);
            }
            
            loadingIndicator.classList.add('hidden');
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
    
    // Navigation controls
    playBtn.addEventListener('click', togglePlayback);
    firstFrameBtn.addEventListener('click', goToFirstFrame);
    prevFrameBtn.addEventListener('click', goToPrevFrame);
    nextFrameBtn.addEventListener('click', goToNextFrame);
    lastFrameBtn.addEventListener('click', goToLastFrame);
    
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
        uploadBtn.textContent = 'Processing...';
        
        // Send request to server
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
    
    // Display detection results
    function displayDetectionResults(data) {
        currentDetections = data.results;
        
        // Generate class color map if needed
        generateClassColors(data.results);
        
        // Display object count in analysis logs
        const totalObjects = data.results.length;
        const resultLog = document.createElement('div');
        resultLog.innerHTML = `<strong>Detection Results:</strong> Found ${totalObjects} objects`;
        
        // Create list of detected objects grouped by class
        const objectList = document.createElement('ul');
        objectList.style.paddingLeft = '15px';
        
        Object.entries(data.class_counts).forEach(([className, count]) => {
            const listItem = document.createElement('li');
            const colorSwatch = document.createElement('span');
            colorSwatch.className = 'detection-color';
            colorSwatch.style.backgroundColor = classColorMap[className];
            
            listItem.appendChild(colorSwatch);
            listItem.appendChild(document.createTextNode(`${className}: ${count}`));
            objectList.appendChild(listItem);
            
            // Update class filter state
            const classFilter = document.getElementById(`filter-${className}`);
            if (classFilter) {
                classFilter.checked = true;
            }
        });
        
        resultLog.appendChild(objectList);
        analysisLogs.appendChild(resultLog);
        
        // Auto-scroll to bottom
        analysisLogs.scrollTop = analysisLogs.scrollHeight;
        
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
        
        // Use different dimensions based on whether we're displaying an image or video
        let displayElement;
        if (!previewImage.classList.contains('hidden')) {
            displayElement = previewImage;
        } else if (!previewVideo.classList.contains('hidden')) {
            displayElement = previewVideo;
        } else {
            return; // Nothing to draw on
        }
        
        // Adjust canvas size to match displayed element
        detectionCanvas.width = displayElement.offsetWidth;
        detectionCanvas.height = displayElement.offsetHeight;
        
        // Calculate scale factors if the original dimensions and display dimensions differ
        const scaleX = detectionCanvas.width / displayElement.naturalWidth || 1;
        const scaleY = detectionCanvas.height / displayElement.naturalHeight || 1;
        
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
            
            // Scale coordinates to match current display size
            const scaledX = x * scaleX;
            const scaledY = y * scaleY;
            const scaledW = w * scaleX;
            const scaledH = h * scaleY;
            
            // Draw bounding box
            if (showBoxes.checked) {
                ctx.strokeStyle = color;
                ctx.lineWidth = 2;
                ctx.strokeRect(scaledX, scaledY, scaledW, scaledH);
            }
            
            // Draw label
            if (showLabels.checked) {
                const label = showConfidence.checked 
                    ? `${className} (${confidence})` 
                    : className;
                
                ctx.font = '14px Arial';
                const textWidth = ctx.measureText(label).width;
                
                ctx.fillStyle = color;
                ctx.fillRect(scaledX, scaledY - 20, textWidth + 10, 20);
                
                ctx.fillStyle = 'white';
                ctx.fillText(label, scaledX + 5, scaledY - 5);
            }
        });
    }
    
    // Load all available classes
    function loadAllClasses() {
        fetch('/get_classes')
            .then(response => response.json())
            .then(data => {
                // Clear existing filters
                classFilters.innerHTML = '';
                
                // Sort classes alphabetically for better usability
                const sortedClasses = data.classes.sort();
                
                // Create a "Select All" checkbox
                const allContainer = document.createElement('div');
                allContainer.className = 'checkbox-container';
                
                const allCheckbox = document.createElement('input');
                allCheckbox.type = 'checkbox';
                allCheckbox.id = 'filter-all';
                allCheckbox.checked = true;
                
                const allLabel = document.createElement('label');
                allLabel.htmlFor = 'filter-all';
                allLabel.appendChild(document.createTextNode('Select All'));
                
                allContainer.appendChild(allCheckbox);
                allContainer.appendChild(allLabel);
                classFilters.appendChild(allContainer);
                
                // Create a container for the class checkboxes
                const classesContainer = document.createElement('div');
                classesContainer.className = 'classes-container';
                
                // "Select All" checkbox functionality
                allCheckbox.addEventListener('change', function() {
                    const checkboxes = classesContainer.querySelectorAll('input[type="checkbox"]');
                    checkboxes.forEach(checkbox => {
                        checkbox.checked = this.checked;
                    });
                    redrawDetections();
                });
                
                // Add class checkboxes
                sortedClasses.forEach(className => {
                    // Generate a random color if not already assigned
                    if (!classColorMap[className]) {
                        const hue = Math.random() * 360;
                        classColorMap[className] = `hsl(${hue}, 70%, 50%)`;
                    }
                    
                    const filterContainer = document.createElement('div');
                    filterContainer.className = 'checkbox-container';
                    
                    const checkbox = document.createElement('input');
                    checkbox.type = 'checkbox';
                    checkbox.id = `filter-${className}`;
                    checkbox.checked = true;
                    checkbox.dataset.class = className;
                    checkbox.addEventListener('change', function() {
                        // Update "Select All" checkbox if needed
                        const allChecked = Array.from(
                            classesContainer.querySelectorAll('input[type="checkbox"]')
                        ).every(cb => cb.checked);
                        
                        allCheckbox.checked = allChecked;
                        redrawDetections();
                    });
                    
                    const label = document.createElement('label');
                    label.htmlFor = `filter-${className}`;
                    
                    const colorSwatch = document.createElement('span');
                    colorSwatch.className = 'detection-color';
                    colorSwatch.style.backgroundColor = classColorMap[className];
                    
                    label.appendChild(colorSwatch);
                    label.appendChild(document.createTextNode(` ${className}`));
                    
                    filterContainer.appendChild(checkbox);
                    filterContainer.appendChild(label);
                    
                    classesContainer.appendChild(filterContainer);
                });
                
                classFilters.appendChild(classesContainer);
            })
            .catch(error => {
                console.error('Error loading classes:', error);
                classFilters.innerHTML = 'Error loading classes';
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
    
    // Initialize webcam
    function initializeWebcam() {
        // Redirect to webcam page
        window.location.href = '/webcam';
    }
    
    // Reset UI state
    function resetUIState() {
        fileUploadContainer.classList.add('hidden');
        previewImage.classList.add('hidden');
        previewVideo.classList.add('hidden');
        loadingIndicator.classList.remove('hidden');
        clearDetectionCanvas();
        
        // Reset file input
        fileInput.value = '';
        fileNameSpan.textContent = 'No file chosen';
        currentFile.textContent = 'Current file: No file selected';
        
        // Reset frame counter
        currentFrame = 0;
        totalFrames = 0;
        frameCounter.textContent = '1 / 0';
        
        // Reset playback
        isPlaying = false;
        playBtn.textContent = '▶';
    }
    
    // Playback controls
    function togglePlayback() {
        if (!previewVideo.classList.contains('hidden')) {
            if (isPlaying) {
                previewVideo.pause();
                playBtn.textContent = '▶';
            } else {
                previewVideo.play();
                playBtn.textContent = '❚❚';
            }
            isPlaying = !isPlaying;
        }
    }
    
    function goToFirstFrame() {
        if (!previewVideo.classList.contains('hidden')) {
            previewVideo.currentTime = 0;
            currentFrame = 0;
            updateFrameCounter();
        }
    }
    
    function goToPrevFrame() {
        if (!previewVideo.classList.contains('hidden')) {
            // Go back 1/30th of a second (assuming 30fps)
            previewVideo.currentTime = Math.max(0, previewVideo.currentTime - (1/30));
            currentFrame = Math.max(0, currentFrame - 1);
            updateFrameCounter();
        }
    }
    
    function goToNextFrame() {
        if (!previewVideo.classList.contains('hidden')) {
            // Go forward 1/30th of a second (assuming 30fps)
            previewVideo.currentTime = Math.min(previewVideo.duration, previewVideo.currentTime + (1/30));
            currentFrame = Math.min(totalFrames - 1, currentFrame + 1);
            updateFrameCounter();
        }
    }
    
    function goToLastFrame() {
        if (!previewVideo.classList.contains('hidden')) {
            previewVideo.currentTime = previewVideo.duration;
            currentFrame = totalFrames - 1;
            updateFrameCounter();
        }
    }
    
    function updateFrameCounter() {
        frameCounter.textContent = `${currentFrame + 1} / ${totalFrames}`;
    }
    
    // Update video timestamp when it's playing
    previewVideo.addEventListener('timeupdate', function() {
        if (isPlaying) {
            currentFrame = Math.floor(previewVideo.currentTime * 30); // Assuming 30fps
            updateFrameCounter();
        }
    });
});