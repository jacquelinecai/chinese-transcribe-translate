from flask import Flask, jsonify, request
import os
from werkzeug.utils import secure_filename

from boxes import main as extract_text_main
from transcribe import main as transcribe_main
# from test_transformer import main as translate_main

app = Flask(__name__)

# Configure upload folder for images
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit file size to 16MB

@app.route("/", methods=['GET'])
def index():
    return ""

@app.route('/api/extract-text', methods=['POST'])
def extract_text():
    """
    Extract text from uploaded images using boxes.py
    """
    print("Request received")
    print(f"Request files: {request.files}")
    print(f"Request files keys: {list(request.files.keys())}")
    # Check if request contains files
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
        # Call the extract_text_main function
        result = extract_text_main()
        return jsonify({'result': result}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/transcribe', methods=['GET'])
def transcribe():
    """
    Transcribe text using transcribe.py
    """
    try:
        # Call the transcribe_main function
        result = transcribe_main()
        return {'result': result}, 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# @app.route('/api/translate', methods=['POST'])
# def translate():
#     """
#     Translate text using test_transformer.py
#     """
#     data = request.json
#     if not data or 'text' not in data:
#         return jsonify({'error': 'No text provided'}), 400
    
#     input_text = data['text']
    
#     try:
#         # Call the translate_main function
#         result = translate_main(input_text)
#         return jsonify({'result': result}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

def allowed_file(filename, allowed_extensions):
    """
    Check if the file extension is allowed
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

if __name__ == '__main__':
    app.run(debug=True)