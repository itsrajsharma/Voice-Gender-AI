from flask import Flask, request, jsonify, render_template
import joblib
import os
from utils import extract_mfcc  

app = Flask(__name__)
model = joblib.load('gender_classifier_model.pkl')

UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']

    if file and allowed_file(file.filename):
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        mfccs = extract_mfcc(filepath)

        if mfccs is not None:
            prediction = model.predict([mfccs])[0]
            return jsonify({"prediction": prediction}), 200
        else:
            return jsonify({"error": "Failed to process the file"}), 500
    else:
        return jsonify({"error": "Invalid file type"}), 400

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'wav', 'mp3', 'flac'}

if __name__ == '__main__':
    app.run(debug=True)

