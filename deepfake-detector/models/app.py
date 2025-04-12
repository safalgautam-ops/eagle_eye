from flask import Flask, request, jsonify
import os
from simple_detector import DeepfakeDetector
import uuid

app = Flask(__name__)
detector = DeepfakeDetector()

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/detect', methods=['POST'])
def detect_deepfake():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
        
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    # Save the uploaded file
    filename = str(uuid.uuid4()) + os.path.splitext(video_file.filename)[1]
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    video_file.save(filepath)
    
    try:
        # Process the video with our detector
        result = detector.predict(filepath)
        # Add file path to the result
        result['filepath'] = filepath
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'OK', 'message': 'Deepfake detection service is running'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)