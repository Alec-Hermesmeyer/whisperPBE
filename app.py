import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import whisper
import torch
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'m4a', 'mp3', 'wav', 'aac', 'flac', 'webm'}
MODEL_NAME = 'base'  # Options: tiny, base, small, medium, large

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load the Whisper model
print("Loading Whisper model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model(MODEL_NAME).to(device)
print(f"Whisper model '{MODEL_NAME}' loaded on {device}.")

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """Endpoint to handle audio file uploads and transcribe them."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request.'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading.'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Perform transcription
        try:
            print(f"Transcribing file: {file_path}")
            result = model.transcribe(file_path, fp16=torch.cuda.is_available())
            transcription = result['text'].strip()
            print(f"Transcription completed: {transcription}")

            # Optionally, delete the uploaded file after transcription
            os.remove(file_path)

            return jsonify({'transcription': transcription}), 200

        except Exception as e:
            print(f"Error during transcription: {e}")
            return jsonify({'error': str(e)}), 500

    else:
        return jsonify({'error': f'Allowed file types are {", ".join(ALLOWED_EXTENSIONS)}.'}), 400

@app.route('/', methods=['GET'])
def index():
    """Simple index route."""
    return "Whisper Transcription API is running."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
