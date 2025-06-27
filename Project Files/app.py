from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import cv2
import os

app = Flask(__name__)
model = load_model("vgg16_model.h5")

# Load and encode labels
train_csv = pd.read_csv("Training_set.csv")
labels = train_csv['label'].unique()
labels.sort()
label_encoder = LabelEncoder()
label_encoder.fit(labels)

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/input')
def input_page():
    return render_template("input.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "❌ No file uploaded"

    files = request.files.getlist("file")
    predictions = []
    filepaths = []

    for file in files:
        if file and file.filename != '':
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            filepaths.append(filepath)

            img = cv2.imread(filepath)
            img = cv2.resize(img, (224, 224))
            img = preprocess_input(img)
            img = np.expand_dims(img, axis=0)

            prediction = model.predict(img)
            pred_class = np.argmax(prediction)
            result = label_encoder.inverse_transform([pred_class])[0]
            predictions.append(result)

    if not predictions:
        return "❌ No valid images uploaded"

    return render_template("output.html", predictions=predictions, filepaths=filepaths)

if __name__ == "__main__":
    app.run(debug=True)