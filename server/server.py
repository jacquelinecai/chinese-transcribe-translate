from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import shutil

from boxes import main as extract_text_main
from transcribe import main as transcribe_main
# from test_transformer import main as translate_main
from translation import main as translate_main

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)
# Configure upload folder for images
UPLOAD_FOLDER = 'uploads'
UPLOAD_CELLS_FOLDER = 'uploaded_extracted_cells'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_CELLS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

@app.route("/", methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/api/extract-cells', methods=['POST'])
def extract_text():
    """
    Extract text from uploaded images using boxes.py
    """
    print("Request received")
    print(f"Request files: {request.files}")
    print(f"Request files keys: {list(request.files.keys())}")
    if 'images' not in request.files:
        return jsonify({'error': 'No images provided'}), 400
    
    files = request.files.getlist('images')
    if not files or files[0].filename == '':
        return jsonify({'error': 'No selected files'}), 400
    
    # Save uploaded files
    file_paths = []
    for file in files:
        if file and allowed_file(file.filename, {'png', 'jpg', 'jpeg'}):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            file_paths.append(file_path)
    
    if not file_paths:
        return jsonify({'error': 'No valid image files provided'}), 400
    
    try:
        result = extract_text_main()
        if os.path.exists(UPLOAD_FOLDER):
            shutil.rmtree(UPLOAD_FOLDER)
            print(f"Directory '{UPLOAD_FOLDER}' and its contents deleted successfully.")
        return jsonify({'result': result}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/transcribe', methods=['GET'])
def transcribe():
    """
    Transcribe text using transcribe.py
    """
    try:
        result = transcribe_main()
        if os.path.exists(UPLOAD_CELLS_FOLDER):
            shutil.rmtree(UPLOAD_CELLS_FOLDER)
            print(f"Directory '{UPLOAD_CELLS_FOLDER}' and its contents deleted successfully.")
        return {'result': result}, 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/translate', methods=['POST'])
def translate():
    """
    Translate text using test_transformer.py
    """
    data = request.json
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    input_text = data['text']
    
    if not isinstance(input_text, list):
        return jsonify({'error': 'Text must be an array of strings'}), 400
    
    if not all(isinstance(item, str) for item in input_text):
        return jsonify({'error': 'All items in the text array must be strings'}), 400

    try:
        result = translate_main(input_text)
        return {'result': result}, 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def allowed_file(filename, allowed_extensions):
    """
    Check if the file extension is allowed
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

if __name__ == '__main__':
    app.run(debug=True)