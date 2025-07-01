import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# ✅ 1. Load your trained model
model = load_model("vgg16_model.h5")

# ✅ 2. Encode labels again (same as used in training)
train_csv = pd.read_csv("Training_set.csv")
labels = train_csv['label'].unique()
labels.sort()
label_encoder = LabelEncoder()
label_encoder.fit(labels)

# ✅ 3. Load and preprocess the image to test
img_path = "test\image_7.jpg"

img = cv2.imread(img_path)
img = cv2.resize(img, (224, 224))
img = preprocess_input(img)
img = np.expand_dims(img, axis=0)

# ✅ 4. Predict
prediction = model.predict(img)
predicted_class = np.argmax(prediction)
class_label = label_encoder.inverse_transform([predicted_class])[0]

print("🦋 Predicted Butterfly Class:", class_label)

