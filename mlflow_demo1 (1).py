# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 21:30:22 2024

@author: retro
"""

import pandas as pd

# Load the dataset
df = pd.read_csv("/Users/andrewprabinrajyesuraj/Downloads/data.csv")
df.head()


#Dropping columns that are not needed
df = df.drop(columns=['id', 'Unnamed: 32'])

#Map the target to binary values: 'M' to 1 (malignant), 'B' to 0 (benign)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Separate features and target datasets
X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=102)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Define the model hyperparameters
params = {
    "solver": "lbfgs",
    "max_iter": 10000,
    "multi_class": "auto",
    "random_state": 8888,
}


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

#train the model
model = LogisticRegression(**params)
model.fit(X_train, y_train)

#Predict and evaluate the model
y_pred = model.predict(X_test)
print("Logistic Regression:")
print(classification_report(y_test, y_pred))

class_report = classification_report(y_test, y_pred,output_dict= True)

#create a metrics which we want to log 

import mlflow

mlflow.set_experiment("cancer_data")
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000/")

with mlflow.start_run():
    mlflow.log_params(params)
    mlflow.log_metrics({
        'accuracy': class_report['accuracy'],
        'recall_class_0': class_report['0']['recall'],
        'recall_class_1': class_report['1']['recall'],
        'f1_score': class_report['macro avg']['f1-score']
        })
    mlflow.sklearn.log_model(model, "Logistic Regression")  
    

