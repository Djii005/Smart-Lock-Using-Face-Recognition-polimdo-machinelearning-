document.addEventListener('DOMContentLoaded', function() {
    // Get DOM elements
    const captureBtn = document.getElementById('captureBtn');
    const trainBtn = document.getElementById('trainBtn');
    const captureForm = document.getElementById('captureForm');
    const startCaptureBtn = document.getElementById('startCaptureBtn');
    const cancelCaptureBtn = document.getElementById('cancelCaptureBtn');
    const userName = document.getElementById('userName');
    const statusMessage = document.getElementById('statusMessage');
    const recognitionInfo = document.getElementById('recognitionInfo');
    const captureCount = document.getElementById('captureCount');
    
    // Update recognition info every second
    setInterval(() => {
        fetch('/recognition_status')
            .then(response => response.json())
            .then(data => {
                if (data.name) {
                    recognitionInfo.textContent = `Recognized: ${data.name} (${data.confidence.toFixed(1)}%)`;
                    recognitionInfo.style.color = data.confidence >= 60 ? '#4CAF50' : '#f44336';
                } else {
                    recognitionInfo.textContent = 'Status: No face detected';
                    recognitionInfo.style.color = '#333';
                }
            })
            .catch(error => {
                console.error('Error fetching recognition status:', error);
            });
    }, 1000);
    
    // Show capture form when capture button is clicked
    captureBtn.addEventListener('click', function() {
        captureForm.style.display = 'block';
        statusMessage.textContent = 'Enter user name and click "Start Capture"';
    });
    
    // Hide capture form when cancel button is clicked
    cancelCaptureBtn.addEventListener('click', function() {
        captureForm.style.display = 'none';
        statusMessage.textContent = 'Capture cancelled.';
    });
    
    // Start face capture when start button is clicked
    startCaptureBtn.addEventListener('click', function() {
        const name = userName.value.trim();
        if (!name) {
            statusMessage.textContent = 'Please enter a user name.';
            return;
        }
        
        captureForm.style.display = 'none';
        statusMessage.textContent = 'Starting face capture... Position your face in the frame.';
        captureCount.textContent = 'Capturing: 0/50';
        
        fetch('/start_capture?name=' + encodeURIComponent(name))
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    captureBtn.disabled = true;
                    trainBtn.disabled = true;
                    
                    // Start checking capture progress
                    const progressCheck = setInterval(() => {
                        fetch('/capture_progress')
                            .then(response => response.json())
                            .then(data => {
                                captureCount.textContent = `Capturing: ${data.count}/50`;
                                
                                if (data.completed) {
                                    clearInterval(progressCheck);
                                    statusMessage.textContent = `Captured ${data.count} images for user ${name}. You can now train the model.`;
                                    captureCount.textContent = '';
                                    captureBtn.disabled = false;
                                    trainBtn.disabled = false;
                                }
                            })
                            .catch(error => {
                                console.error('Error checking capture progress:', error);
                                clearInterval(progressCheck);
                                captureBtn.disabled = false;
                                trainBtn.disabled = false;
                                statusMessage.textContent = 'Error during face capture. Please try again.';
                            });
                    }, 500);
                } else {
                    statusMessage.textContent = 'Error starting face capture.';
                }
            })
            .catch(error => {
                console.error('Error starting face capture:', error);
                statusMessage.textContent = 'Error communicating with the server. Please try again.';
            });
    });
    
    // Train the model when train button is clicked
    trainBtn.addEventListener('click', function() {
        statusMessage.textContent = 'Training face recognition model...';
        trainBtn.disabled = true;
        
        fetch('/train_model')
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    statusMessage.textContent = `Training completed successfully with ${data.faces} images of ${data.users} users.`;
                } else {
                    statusMessage.textContent = 'Error training model: ' + data.message;
                }
                trainBtn.disabled = false;
            })
            .catch(error => {
                console.error('Error training model:', error);
                statusMessage.textContent = 'Error communicating with the server. Please try again.';
                trainBtn.disabled = false;
            });
    });
});