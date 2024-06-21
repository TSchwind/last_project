import pandas as pd
import numpy as np
import pickle
import pytest
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from starter.ml.data import process_data
from starter.ml.model import train_model, inference, compute_model_metrics


@pytest.fixture
def data():
    "Function to read input data."
    df = pd.read_csv('starter/data/census.csv')

    return df


@pytest.fixture
def model():
    "Function to load trained model."
    model = pickle.load(open('starter/model/clf_model.pkl', "rb"))

    return model


@pytest.fixture
def encoder():
    "Function to load used encoder."
    encoder = pickle.load(open('starter/model/encoder.pkl', "rb"))

    return encoder


@pytest.fixture
def label_binarizer():
    "Function to load used encoder."
    lb = pickle.load(open('starter/model/label_binarizer.pkl', "rb"))

    return lb


@pytest.fixture
def get_train_test_data(data):
    train, test = train_test_split(data, test_size=0.20)
    return train, test


@pytest.fixture
def get_train_test(get_train_test_data, encoder, label_binarizer):
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
    X_train, y_train, _, _ = process_data(
        get_train_test_data[0], categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=label_binarizer
    )

    X_test, y_test, _, _ = process_data(
        get_train_test_data[1], categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=label_binarizer
    )
    return X_train, y_train, X_test, y_test


def test_data_shape(data):
    """
    Test for not having null values.
    """
    print(data)
    assert data.shape == data.dropna().shape


def test_train_model(get_train_test):
    """
    Test if train_model returns a DecisionTreeClassifier.
    """
    trained_model = train_model(get_train_test[0], get_train_test[1])
    assert isinstance(trained_model, DecisionTreeClassifier)


def test_inference(model, get_train_test):
    """
    Test if inference returns numpy.ndarray
    """

    prediction = inference(model, get_train_test[2])
    assert isinstance(prediction, np.ndarray)


def test_compute_model_metrics(model, get_train_test):
    """
    Test if compute_model_metrics returns floats.
    """
    prediction = inference(model, get_train_test[2])
    precision, recall, fbeta = compute_model_metrics(
        get_train_test[3], prediction)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)
