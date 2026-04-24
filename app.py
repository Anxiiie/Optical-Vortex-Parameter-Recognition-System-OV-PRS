"""
Flask web server for Laguerre-Gaussian optical vortex recognition
"""

import os
import cv2
import numpy as np
import threading
import time
from flask import Flask, render_template, request, jsonify, Response
from werkzeug.utils import secure_filename
from neural_network import VortexRecognizer

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 512 * 1024 * 1024  # 512MB max file size

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables
recognizer = None
camera = None
camera_running = False
recognition_active = False
recognition_results = []
recognition_lock = threading.Lock()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

def allowed_file(filename):
    """Check allowed file extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_camera_frame():
    """Get frame from camera"""
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)

    ret, frame = camera.read()
    if ret:
        return frame
    return None

def recognition_loop():
    """Background recognition loop from camera"""
    global camera_running, recognition_active, recognition_results

    while camera_running and recognition_active:
        frame = get_camera_frame()
        if frame is not None:
            # Recognition
            result = recognizer.recognize(frame)
            if result:
                result['filename'] = f"camera_{int(time.time())}.jpg"
                with recognition_lock:
                    recognition_results.append(result)

        time.sleep(0.1)  # ~10 FPS

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/load_model', methods=['POST'])
def load_model():
    """Load pre-trained model"""
    global recognizer

    if 'model_file' not in request.files:
        return jsonify({'success': False, 'message': 'Model file not uploaded'})

    file = request.files['model_file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No file selected'})

    if file:
        filename = secure_filename(file.filename)
        model_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(model_path)

        # Load model
        recognizer = VortexRecognizer(model_path)

        if recognizer.model is not None:
            return jsonify({'success': True, 'message': 'Model loaded successfully'})
        else:
            return jsonify({'success': False, 'message': 'Model loading error'})

@app.route('/start_camera', methods=['POST'])
def start_camera():
    """Start camera"""
    global camera, camera_running, recognition_active, recognition_results

    if recognizer is None:
        return jsonify({'success': False, 'message': 'Model not loaded'})

    if camera is None:
        camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        return jsonify({'success': False, 'message': 'Failed to open camera'})

    camera_running = True
    recognition_active = True
    recognition_results = []

    # Start background recognition thread
    recognition_thread = threading.Thread(target=recognition_loop)
    recognition_thread.daemon = True
    recognition_thread.start()

    return jsonify({'success': True, 'message': 'Camera started'})

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    """Stop camera"""
    global camera_running, recognition_active, camera

    camera_running = False
    recognition_active = False

    if camera is not None:
        camera.release()
        camera = None

    return jsonify({'success': True, 'message': 'Camera stopped'})

@app.route('/camera_feed')
def camera_feed():
    """Video stream from camera"""
    def generate():
        global camera
        while camera_running and camera is not None:
            ret, frame = camera.read()
            if ret:
                ret, jpeg = cv2.imencode('.jpg', frame)
                frame_bytes = jpeg.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
            time.sleep(0.03)

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    """Upload image from disk"""
    global recognizer, recognition_results

    if recognizer is None:
        return jsonify({'success': False, 'message': 'Model not loaded'})

    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'File not uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No file selected'})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Recognition
        result = recognizer.recognize(filepath)
        if result:
            result['filename'] = filename
            with recognition_lock:
                recognition_results.append(result)
            return jsonify({'success': True, 'result': result})
        else:
            return jsonify({'success': False, 'message': 'Recognition error'})

    return jsonify({'success': False, 'message': 'Invalid file format'})

@app.route('/upload_images', methods=['POST'])
def upload_images():
    """Upload multiple images"""
    global recognizer, recognition_results

    if recognizer is None:
        return jsonify({'success': False, 'message': 'Model not loaded'})

    files = request.files.getlist('files')
    if not files:
        return jsonify({'success': False, 'message': 'Files not uploaded'})

    results = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Recognition
            result = recognizer.recognize(filepath)
            if result:
                result['filename'] = filename
                with recognition_lock:
                    recognition_results.append(result)
                results.append(result)

    return jsonify({'success': True, 'count': len(results), 'results': results})

@app.route('/get_results')
def get_results():
    """Get recognition results"""
    global recognition_results

    with recognition_lock:
        return jsonify({'results': recognition_results})

@app.route('/stop_recognition', methods=['POST'])
def stop_recognition():
    """Stop recognition"""
    global recognition_active

    recognition_active = False
    return jsonify({'success': True, 'message': 'Recognition stopped'})

@app.route('/clear_results', methods=['POST'])
def clear_results():
    """Clear results"""
    global recognition_results

    with recognition_lock:
        recognition_results = []

    return jsonify({'success': True, 'message': 'Results cleared'})

@app.route('/export_results', methods=['POST'])
def export_results():
    """Export results to CSV with language support"""
    global recognition_results

    data = request.get_json()
    language = data.get('language', 'ru') if data else 'ru'

    # CSV headers for different languages
    headers = {
        'en': 'filename,n,m,TC',
        'ru': 'имя_файла,n,m,TC',
        'zh': '文件名,n,m,TC',
        'ja': 'ファイル名,n,m,TC'
    }

    with recognition_lock:
        if not recognition_results:
            return jsonify({'success': False, 'message': 'No results to export'})

        csv_content = headers.get(language, 'filename,n,m,TC') + "\n"
        for result in recognition_results:
            csv_content += f"{result['filename']},{result['n']},{result['m']},{result['TC']}\n"

    return jsonify({'success': True, 'csv': csv_content})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True, threaded=True)
