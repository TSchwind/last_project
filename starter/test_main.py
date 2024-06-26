import json
from fastapi.testclient import TestClient

from main import app  # noqa

client = TestClient(app)


def test_get():
    response = client.get("/")
    assert response.json()[
        "greeting"] == "Welcome to the last udacity MLDevOps Training project!"
    assert response.status_code == 200


def test_post_case1():
    body = json.dumps({
        "columns": [
            "age",
            "workclass",
            "fnlgt",
            "education",
            "education-num",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
            "native-country"
        ],
        "data": [
            [
                39,
                "State-gov",
                77516,
                "Bachelors",
                13,
                "Never-married",
                "Adm-clerical",
                "Not-in-family",
                "White",
                "Male",
                2174,
                0,
                40,
                "United-States"
            ]
        ],
        "encoder_path": "starter/model/encoder.pkl",
        "label_binarizer_path": "starter/model/label_binarizer.pkl",
        "path_model": "starter/model/clf_model.pkl"
    })
    response = client.post('/inference/', data=body)
    assert response.status_code == 200
    assert response.json()["prediction"] == 0


def test_post_case2():
    body = json.dumps({
        "columns": [
            "age",
            "workclass",
            "fnlgt",
            "education",
            "education-num",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
            "native-country"
        ],
        "data": [
            [52, "Self-emp-not-inc", 209642, "HS-grad", 9, "Married-civ-spouse",
                "Exec-managerial", "Husband", "White", "Male", 0, 0, 45, "United-States"]
        ],
        "encoder_path": "starter/model/encoder.pkl",
        "label_binarizer_path": "starter/model/label_binarizer.pkl",
        "path_model": "starter/model/clf_model.pkl"
    })
    response = client.post('/inference/', data=body)
    assert response.status_code == 200
    assert response.json()["prediction"] == 1
