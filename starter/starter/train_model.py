# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import os
import pickle
import pandas as pd
from ml.data import process_data
from ml.model import train_model

# Add code to load in the data.
data = pd.read_csv('starter/data/census.csv')

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)


# Train and save a model.
trained_model = train_model(X_train, y_train)


# write the trained model to your workspace in a file called trainedmodel.pkl
if not os.path.exists("starter/model"):
    os.mkdir("starter/model")
with open(os.path.join("starter/model", "clf_model.pkl"), "wb") as f:
    pickle.dump(trained_model, f)
