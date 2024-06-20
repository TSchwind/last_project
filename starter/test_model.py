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
def get_train_x_y(get_train_test_data):
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
        get_train_test_data[0], categorical_features=cat_features, label="salary", training=True
    )
    return X_train, y_train, encoder, lb


@pytest.fixture
def get_test_x_y(get_train_test_data, get_train_x_y):
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

    X_test, y_test, encoder, lb = process_data(
        get_train_test_data[1], categorical_features=cat_features, label="salary", training=False, encoder=get_train_x_y[2], lb=get_train_x_y[3]
    )
    return X_test, y_test, encoder, lb


def test_data_shape(data):
    """
    Test for not having null values.
    """
    print(data)
    assert data.shape == data.dropna().shape


def test_train_model(get_train_x_y):
    """
    Test if train_model returns a DecisionTreeClassifier.
    """
    trained_model = train_model(get_train_x_y[0], get_train_x_y[1])
    assert isinstance(trained_model, DecisionTreeClassifier)


def test_inference(model, get_test_x_y):
    """
    Test if inference returns numpy.ndarray
    """

    prediction = inference(model, get_test_x_y[0])
    assert isinstance(prediction, np.ndarray)


def test_compute_model_metrics(model, get_test_x_y):
    """

    """
    prediction = inference(model, get_test_x_y[0])
    precision, recall, fbeta = compute_model_metrics(
        get_test_x_y[1], prediction)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)
