# Waste_Image_Classifier
# ♻️ Waste Image Classification System
Improper waste segregation is a major environmental issue. This project aims to leverage deep learning-based image classification to assist in automated waste sorting. The system will take an input image and predict its waste category, helping improve recycling efficiency and sustainability efforts.

## 📌 Problem Statement
This project aims to build a deep learning-based computer vision system that classifies waste materials from images into categories such as plastic, paper, metal, and glass. The system also includes a web interface for real-time image upload and prediction.

---

## 🎯 Objectives
- Develop an image classification model using transfer learning
- Achieve at least 85% accuracy on validation data
- Build a web application for user interaction
- Extend the system to object detection (future scope)

---

## 🧠 Approach
- Dataset: Waste classification dataset (TrashNet/Kaggle)
- Model: Pretrained ResNet18 / MobileNetV2
- Framework: PyTorch
- Deployment: Streamlit Web App

---

## 🏗️ Project Structure
```
waste-classifier/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── src/
│   ├── train.py
│   ├── predict.py
│   ├── utils.py
│
├── notebooks/
│   ├── EDA.ipynb
│
├── app/
│   ├── app.py   # Streamlit later
│
├── models/
│   ├── model.pth
│
├── docs/
│
├── requirements.txt
├── README.md
```

---

## ⚙️ Installation
```bash
git clone <this-repo-link>
cd waste-classifier
pip install -r requirements.txt
```

## Train Model
```bash
python src/train.py
```

## Run App
```bash
streamlit run app/app.py
```
