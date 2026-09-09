import os
import sys
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from gevent.pywsgi import WSGIServer

# Ensure project root is in sys.path
FILE = Path(__file__).resolve()
ROOT = FILE.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from util import base64_to_pil
from src.detector import ANPRDetector

# Initialize Flask App
app = Flask(__name__)

# Ensure upload directory exists
UPLOAD_DIR = ROOT / "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Lazy Load ANPR Detector Instance
detector_instance = None

def get_detector():
    global detector_instance
    if detector_instance is None:
        detector_instance = ANPRDetector(weights_path=str(ROOT / "weights" / "best.pt"))
    return detector_instance

@app.route('/', methods=['GET'])
def index():
    """Renders the primary home page interface."""
    return render_template('newin.html')

@app.route('/english_detection', methods=['GET'])
def english_detection():
    """Renders the English ANPR interface."""
    return render_template('index.html')

@app.route('/marathi_detection', methods=['GET'])
def marathi_detection():
    """Renders the Marathi regional ANPR interface."""
    return render_template('index_for_marathi.html')

@app.route('/about', methods=['GET'])
def about():
    """Renders the about project documentation view."""
    return render_template('about.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Processes image post payload for English ANPR."""
    if request.method == 'POST':
        try:
            img = base64_to_pil(request.json)
            image_path = UPLOAD_DIR / "image.png"
            img.save(str(image_path))

            detector = get_detector()
            res = detector.detect_and_recognize(str(image_path), lang='eng')
            return jsonify(result=str(res.get("detected_plate", "Not Detected")))
        except Exception as e:
            return jsonify(result="Error processing image", error=str(e)), 500
            
    return jsonify(status="ready", mode="English ANPR")

@app.route('/predict_marathi', methods=['GET', 'POST'])
def predict_marathi():
    """Processes image post payload for Marathi regional ANPR."""
    if request.method == 'POST':
        try:
            img = base64_to_pil(request.json)
            image_path = UPLOAD_DIR / "image.png"
            img.save(str(image_path))

            detector = get_detector()
            res = detector.detect_and_recognize(str(image_path), lang='mr')
            return jsonify(result=str(res.get("detected_plate", "Not Detected")))
        except Exception as e:
            return jsonify(result="Error processing image", error=str(e)), 500

    return jsonify(status="ready", mode="Marathi ANPR")

@app.route('/api/v1/predict', methods=['POST'])
def api_predict():
    """
    RESTful API Endpoint for License Plate Recognition.
    Expects JSON body: {"image_base64": "<base64_string>", "language": "eng" | "mr"}
    """
    data = request.get_json(force=True, silent=True) or {}
    image_base64 = data.get("image_base64")
    lang = data.get("language", "eng")

    if not image_base64:
        return jsonify({"status": "error", "message": "Missing 'image_base64' field in payload."}), 400

    try:
        img = base64_to_pil(image_base64)
        image_path = UPLOAD_DIR / "api_upload.png"
        img.save(str(image_path))

        detector = get_detector()
        response_data = detector.detect_and_recognize(str(image_path), lang=lang)
        return jsonify(response_data), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    print("Serving ANPR Application on http://0.0.0.0:5000 ...")
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
