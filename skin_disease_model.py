import os
import zipfile
import pandas as pd
import shutil
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Step 1: Extract dataset zip file
zip_path = 'ISIC-images.zip'
extract_path = 'ISIC_images'

if os.path.exists(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("✅ Dataset extracted successfully.")
else:
    print("⚠️ ISIC-images.zip not found. Please download the dataset manually.")

# Step 2: Organize images by labels based on metadata
csv_file = 'HAM10000_metadata.csv'
image_folder = 'ISIC_dataset'          # Folder containing all image files
output_dir = 'ISIC_dataset_labeled'    # Output directory for labeled folders

if os.path.exists(csv_file):
    df = pd.read_csv(csv_file)
    os.makedirs(output_dir, exist_ok=True)

    for index, row in df.iterrows():
        label = row['diagnosis']
        img_name = row['isic_id'] + '.jpg'
        src_path = os.path.join(image_folder, img_name)
        label_folder = os.path.join(output_dir, label)
        os.makedirs(label_folder, exist_ok=True)

        dst_path = os.path.join(label_folder, img_name)
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
        else:
            print(f"Image not found: {img_name}")
else:
    print("⚠️ Metadata CSV not found. Please place HAM10000_metadata.csv in this folder.")

# Step 3: Preprocess images
def preprocess_images(data_dir, image_size=(128, 128)):
    data, labels = [], []
    classes = os.listdir(data_dir)

    for label in classes:
        folder_path = os.path.join(data_dir, label)
        for img_name in tqdm(os.listdir(folder_path), desc=f"Processing {label}"):
            img_path = os.path.join(folder_path, img_name)
            try:
                img = cv2.imread(img_path)
                img = cv2.resize(img, image_size)
                img = img / 255.0
                data.append(img)
                labels.append(label)
            except:
                continue
    return np.array(data), np.array(labels)

if os.path.exists(output_dir):
    X, y = preprocess_images(output_dir)
else:
    print("⚠️ Labeled dataset not found. Please prepare it first.")

# Step 4: Encode labels and split data
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42
)

# Step 5: Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=X_train.shape[1:]),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(y_train.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Step 6: Train the model
history = model.fit(X_train, y_train, validation_split=0.2, epochs=10, batch_size=32)

# Step 7: Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"✅ Test Accuracy: {accuracy * 100:.2f}%")

# Step 8: Save the model
model.save('skin_disease_model.h5')
print("💾 Model saved as skin_disease_model.h5")
