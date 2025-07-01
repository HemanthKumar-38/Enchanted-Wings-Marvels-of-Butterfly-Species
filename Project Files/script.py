import pandas as pd
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint  # ✅ ADD THIS

# ✅ Step 2: Load CSVs
train_csv = pd.read_csv("Training_set.csv")
test_csv = pd.read_csv("Testing_set.csv")

print("Train CSV shape:", train_csv.shape)
print("Test CSV shape:", test_csv.shape)
print(train_csv.head())

# ✅ Step 3: Load & Resize Images
image_size = (224, 224)
X = []
y = []

for index, row in train_csv.iterrows():
    img_path = os.path.join("train", row['filename'])
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, image_size)
        X.append(img)
        y.append(row['label'])
    else:
        print(f"❌ File not found: {img_path}")

X = np.array(X)
y = np.array(y)

print("✅ Loaded images:", X.shape)
print("✅ Labels:", y.shape)

# ✅ Step 4: Encode Labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

X_train, X_val, y_train, y_val = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded)

print("✅ X_train:", X_train.shape)
print("✅ y_train:", y_train.shape)
print("✅ X_val:", X_val.shape)
print("✅ y_val:", y_val.shape)

# ✅ Step 5: Load VGG16 base model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(75, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy', metrics=['accuracy'])

# ✅ Step 6: Add checkpoint to save vgg16 model
checkpoint = ModelCheckpoint("vgg16_model.h5", save_best_only=True, monitor="val_accuracy", mode="max")

# ✅ Step 7: Train with checkpoint
history = model.fit(X_train, y_train,
                    epochs=5,
                    batch_size=32,
                    validation_data=(X_val, y_val),
                    callbacks=[checkpoint])  # 🟢 USE CALLBACK HERE

print("✅ Model training complete. Best model saved as 'vgg16_model.h5'")
# 1️⃣ Training code - VGG16, compile, model.fit()
# 2️⃣ Save model — model.save("vgg16_model.h5")

print("✅ Model training complete. Best model saved as 'vgg16_model.h5'")

