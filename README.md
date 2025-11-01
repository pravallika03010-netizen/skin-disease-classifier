# 🧠 Skin Disease Classifier (CNN)

This project classifies **skin disease images** using a **Convolutional Neural Network (CNN)** built with TensorFlow and Keras.

## 📦 Dataset

You will need to download the **HAM10000 dataset** manually from Kaggle:  
👉 [HAM10000 Skin Lesion Dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

Please download and place these files in your project folder:
- `ISIC-images.zip` → contains all image files  
- `HAM10000_metadata.csv` → contains image labels and metadata

## 🚀 How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Place your dataset files (`ISIC-images.zip` and `HAM10000_metadata.csv`) in the same folder.

3. Run the script:
   ```bash
   python skin_disease_model.py
   ```

The script will:
- Extract the dataset
- Organize images into labeled folders
- Train a CNN model
- Save the trained model as `skin_disease_model.h5`

## 🧩 Requirements
See `requirements.txt` for details.

## 🧑‍💻 Author
Created for educational and research purposes to practice deep learning image classification.
