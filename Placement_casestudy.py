#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib   

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

#Step 1: Load the dataset

print("Step 1: Load the dataset")

data = pd.read_csv("placement_data.csv")

print("Dataset loaded successfully!")
print(data)

#Step 2: Understand the data
print("\nStep 2: Understand the data")
print("First 5 rows of the dataset:")
print(data.head())

print("\nDataset Shape:")
print(data.shape)

print("\nColumn Names:")
print(data.columns)

print("statistical summary of the dataset:")
print(data.describe())

print("\nChecking for missing values:")
print(data.isnull().sum())

#Step 3: Preprocess the data
print("\nStep 3: spliting input and output data")

x = data[["Aptitude","Coding","Communication","Academics","Internship"]]

y = data["Placed"]

print("\nInput features (x):")

print(x.head())

print("\nTarget variable (y):")
print(y.head())

#Step 4: Split the data into training and testing sets

print("\nStep 4: Split the data into training and testing sets")
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=42)

print("\nTraining set shape:",x_train.shape)
print("Testing input shape:",x_test.shape)
print("Training output shape:",y_train.shape)
print("Testing output shape:",y_test.shape)


#Step 5: Feature Scaling

print("\nStep 5: Feature Scaling")

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
print("\n scaled training sample data:")
print(x_train_scaled[:5])

#Step 6: Create an FNN model

print("\nStep 6: Create an FNN model")
model = MLPClassifier(hidden_layer_sizes=(8,4),activation='relu',solver='adam',max_iter=1000,random_state=42)
print(model)

#Step 7: Train the model
print("\nStep 7: Train the model")
model.fit(x_train_scaled,y_train)
print("\nModel trained successfully!")

#Step 8: Make predictions on test data
print("\nStep 8: Make predictions on test data")

y_pred = model.predict(x_test_scaled)

print("\nActual output:")
print(y_test.values)

print("\nPredicted output:")
print(y_pred)

#Step 9: Evaluate the model
print("\nStep 9: Evaluate the model")

accuracy = accuracy_score(y_test,y_pred)
print("\nAccuracy:",accuracy)

cm = confusion_matrix(y_test,y_pred)
print("\nConfusion Matrix:")
print(cm)

report = classification_report(y_test,y_pred)
print("\nClassification Report:")
print(report)

#Step 10: Predict probabilities 
print("\nStep 10: Predict probabilities")

y_prob = model.predict_proba(x_test_scaled)
print("\nPredicted probabilities:")
print(y_prob)

#Step 11: Save the model
print("\nStep 11: Save the model")
joblib.dump(model,"placement_model.pkl")

joblib.dump(scaler,"scaler.pkl")

print("model save as: placement_model.pkl")
print("scaler save as: scaler.pkl")

#Step 12: Load the model and make a prediction

print("\nStep 12: Load the model and make a prediction")
loaded_model = joblib.load("placement_model.pkl")
loaded_scaler = joblib.load("scaler.pkl")

print("\nLoaded model and scaler successfully!")

# step 13: New data for prediction
print("\n step 13: New data for prediction:")

new_data = pd.DataFrame([[70,72,75,74,1]], columns=["Aptitude","Coding","Communication","Academics","Internship"])  # Example input data
print("\nNew data:")
print(new_data)

new_data_scaled = loaded_scaler.transform(new_data)

new_prediction = loaded_model.predict(new_data_scaled)
new_prediction_proba = loaded_model.predict_proba(new_data_scaled)
print("\nPredicted class for new data:",new_prediction[0])
print("\nPredicted probabilities for new data:",new_prediction_proba[0])
if new_prediction[0] == 1:
    print("\nThe student is placed.")
else:
    print("\nThe student is not placed.")
    
print("preiction Probability for new data:",new_prediction_proba)

#Step 14: Visualize the results

print("\nStep 14: Visualize the results")

placement_counts = data["Placed"].value_counts()

plt.figure(figsize=(7,5))
plt.bar(["Not Placedd","Placed"],placement_counts.values)
plt.title("Placement Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.grid(True)
plt.show()

# Step 15 : Graph - Aptitude vs coding 

plt.figure(figsize=(8,6))
for i in range(len(data)):
    if data["Placed"][i] == 1:
        plt.scatter(data["Aptitude"][i], data["Coding"][i], marker='o', label="Placed" if i == 4 else "")
    else:
        plt.scatter(data["Aptitude"][i], data["Coding"][i], marker='x', label="Not Placed" if i == 0 else "")
        
plt.title("Aptitude vs Coding")
plt.xlabel("Aptitude Score")
plt.ylabel("Coding Score")
plt.legend()
plt.grid(True)
plt.show()


# Step 16 : Graph 3 - Trainging loss cruve

plt.figure(figsize=(8,5))
plt.plot(model.loss_curve_)
plt.title("Traning Loss Curve")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.grid("True")
plt.show()


# Step 17 : Actual vs Predicted

comparison = pd.DataFrame({"Actual":y_test.values,"Predicted":y_pred})

plt.figure(figsize=(8,5))
plt.plot(comparison["Actual"].values,marker='o',label="Actual")
plt.plot(comparison["Predicted"].values,marker='s',label="Predicted")
plt.title("Actual vs Predicted Placement")
plt.xlabel("Test Sample Index")
plt.ylabel("Class")
plt.legend()
plt.grid(True)
plt.show()

print("\n Project execution completed successfully")



