📊 Placement Prediction System using Machine Learning
🚀 Project Description

This project predicts whether a student will be placed or not placed using Machine Learning.
It uses a Feedforward Neural Network (FNN) implemented with MLPClassifier.

The model analyzes student performance based on multiple factors and provides:


Prediction (Placed / Not Placed)
Probability score
🧠 Features
Data preprocessing & analysis
Feature scaling using StandardScaler
Neural Network model (MLPClassifier)
Model evaluation (Accuracy, Confusion Matrix, Classification Report)
Model saving & loading using Joblib
Data visualization using Matplotlib
New data prediction
🛠️ Technologies Used
Python
Pandas
NumPy
Matplotlib
Scikit-learn
Joblib

📁 Project Structure
placement-prediction/
│
├── placement_data.csv        # Dataset
├── main.py                   # Main Python script
├── placement_model.pkl       # Saved ML model
├── scaler.pkl                # Saved scaler
├── requirements.txt          # Dependencies
├── README.md                 # Project documentation
└── .gitignore
📊 Dataset Information

The dataset contains the following features:

Feature	Description
Aptitude	Aptitude score
Coding	Coding skill score
Communication	Communication skill
Academics	Academic performance
Internship	Internship experience (0/1)
Placed	Target (0 = Not Placed, 1 = Placed)
