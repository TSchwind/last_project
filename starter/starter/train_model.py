# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import os
import pickle
import pandas as pd
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics

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


# write the trained model to your workspace in a file called clf_model.pkl
if not os.path.exists("starter/model"):
    os.mkdir("starter/model")
with open(os.path.join("starter/model", "clf_model.pkl"), "wb") as f:
    pickle.dump(trained_model, f)
with open(os.path.join("starter/model", "encoder.pkl"), "wb") as f:
    pickle.dump(encoder, f)
with open(os.path.join("starter/model", "label_binarizer.pkl"), "wb") as f:
    pickle.dump(lb, f)

# Prediction on whole test data
prediction = inference(trained_model, X_test)
model_precision, mdoel_recall, model_fbeta = compute_model_metrics(
    y_test, prediction)
print("Model performance: ")
print(
    f"precision: {model_precision}, recall: {mdoel_recall}, fbeta: {model_fbeta}")
print("")


def compute_model_performance_of_slice(test, cat_features, encoder, lb, slice_cat):
    with open('slice_output.txt', 'a') as file:
        # file.write("Model performance: \n")
        # file.write(
        #     f"precision: {model_precision}, recall: {mdoel_recall}, fbeta: {model_fbeta}\n")
        # Prediction on test data slices
        # for cat_feature in cat_features:
        print("")
        print(f"Feature {slice_cat}")
        file.write(f"Feature {slice_cat}\n")
        for slice in test[slice_cat].unique():
            X_test, y_test, encoder, lb = process_data(
                test[test[slice_cat] ==
                     slice], categorical_features=cat_features, label="salary",
                training=False, encoder=encoder, lb=lb)
            prediction = inference(trained_model, X_test)
            slice_precision, slice_recall, slice_fbeta = compute_model_metrics(
                y_test, prediction)
            print(f"Model performance for {slice}: ")
            print(
                f"precision: {slice_precision}, recall: {slice_recall}, fbeta: {slice_fbeta}")

            file.write(f"Model performance for {slice}:\n")
            file.write(
                f"precision: {slice_precision}, recall: {slice_recall}, fbeta: {slice_fbeta}\n")
        file.write("\n")


for slice_cat in cat_features:
    compute_model_performance_of_slice(
        test, cat_features, encoder, lb, slice_cat)
