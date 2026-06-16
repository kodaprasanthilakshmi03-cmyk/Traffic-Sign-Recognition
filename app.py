from __future__ import division, print_function
import csv,os
import numpy as np
import cv2
import tensorflow as tf
import pandas as pd
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import send_from_directory
from datetime import datetime
from flask import Flask, request, render_template, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
from utils.model_summary import get_model_summary
# Flask app setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
# Load trained model
MODEL_PATH = 'model/model.h5'
model = load_model(MODEL_PATH)
LABELS_FILE = 'model/labels.csv'
DATASET_DIR = 'Dataset'
CSV_FILE = 'history.csv'
# Class names mapping (0 to 42)
def getClassName(classNo):
    classes = [
        'Speed Limit 20 km/h', 'Speed Limit 30 km/h', 'Speed Limit 50 km/h', 'Speed Limit 60 km/h',
        'Speed Limit 70 km/h', 'Speed Limit 80 km/h', 'End of Speed Limit 80 km/h', 'Speed Limit 100 km/h',
        'Speed Limit 120 km/h', 'No passing', 'No passing for vehicles over 3.5 metric tons',
        'Right-of-way at the next intersection', 'Priority road', 'Yield', 'Stop', 'No vehicles',
        'Vehicles over 3.5 metric tons prohibited', 'No entry', 'General caution',
        'Dangerous curve to the left', 'Dangerous curve to the right', 'Double curve', 'Bumpy road',
        'Slippery road', 'Road narrows on the right', 'Road work', 'Traffic signals', 'Pedestrians',
        'Children crossing', 'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing',
        'End of all speed and passing limits', 'Turn right ahead', 'Turn left ahead', 'Ahead only',
        'Go straight or right', 'Go straight or left', 'Keep right', 'Keep left', 'Roundabout mandatory',
        'End of no passing', 'End of no passing by vehicles over 3.5 metric tons'
    ]
    return classes[classNo] if 0 <= classNo < len(classes) else "Unknown"
# Image preprocessing
def grayscale(img): return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
def equalize(img): return cv2.equalizeHist(img)
def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255.0
    return img
# Predict function
def model_predict(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = np.asarray(img)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    img = img.reshape(1, 32, 32, 1)
    predictions = model.predict(img)
    classIndex = np.argmax(predictions, axis=1)[0]
    label = getClassName(classIndex)
    return label
# Route: Home Page
@app.route('/', methods=['GET'])
def index():
    return render_template('home.html')
# Route: Prediction
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file_path = os.path.join('static/uploads', filename)
        file.save(file_path)
        predicted_label = model_predict(file_path)  # Your prediction function
        # ✅ Append prediction to history.csv
        with open(CSV_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([file_path, predicted_label, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
        return render_template('predict.html', prediction=predicted_label, filename=filename)
    return render_template('predict.html')
# Load labels.csv into a list of dicts
def load_labels():
    df = pd.read_csv(LABELS_FILE)
    return df.to_dict(orient='records')
@app.route('/dataset')
def dataset_view():
    labels = load_labels()
    selected_class_id = request.args.get('class_id')
    image_paths = []
    selected_label = None
    total_images = 0
    if selected_class_id is not None and selected_class_id != "":
        folder_path = os.path.join(DATASET_DIR, str(selected_class_id))
        if os.path.exists(folder_path):
            image_filenames = os.listdir(folder_path)
            image_paths = [f"/dataset_image/{selected_class_id}/{img}" for img in image_filenames]
            print("Images to display:", image_paths)

            total_images = len(image_filenames)

        for label in labels:
            if str(label['ClassId']) == selected_class_id:
                selected_label = label
                break
    return render_template('dataset.html',
                           labels=labels,
                           selected_class_id=selected_class_id,
                           image_paths=image_paths,
                           selected_label=selected_label,
                           total_images=total_images)
# Serve images dynamically from Dataset/<class_id>/ folder
@app.route('/dataset_image/<class_id>/<filename>')
def serve_image(class_id, filename):
    directory = os.path.join(DATASET_DIR, class_id)
    print("Serving image from:", directory, filename)
    return send_from_directory(directory, filename)
@app.route('/model/<path:filename>')
def model_files(filename):
    return send_from_directory('model', filename)
@app.route("/model")
def model_page():
    model_summary = get_model_summary()
    return render_template("model.html", model_summary=model_summary)
@app.route("/history")
def history():
    predictions = []
    if os.path.exists("history.csv"):
        with open("history.csv", "r", newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                full_path = row["Image"].replace("\\", "/")
                # Extract only after 'static/', e.g., 'uploads/xyz.jpg'
                if 'static/' in full_path:
                    relative_path = full_path.split("static/")[1]
                else:
                    relative_path = full_path  # fallback
                predictions.append({
                    "image_path": relative_path,
                    "predicted_class": row["PredictedClass"],
                    "timestamp": row["Timestamp"]
                })
    return render_template("history.html", predictions=predictions)
@app.route('/delete_row', methods=['POST'])
def delete_row():
    index = int(request.form['index'])
    rows = []
    with open(CSV_FILE, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    if 0 <= index < len(rows):
        rows.pop(index)
    with open(CSV_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    return redirect(url_for('history'))
@app.route('/clear_history', methods=['POST'])
def clear_history():
    with open(CSV_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Image', 'Class', 'Timestamp'])
    return redirect(url_for('history'))
# Run server
if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(host='0.0.0.0', port=5001, debug=True)
