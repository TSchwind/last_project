import requests
import json

data = {
    "path_model": "starter/model/clf_model.pkl",
    "encoder_path": "starter/model/encoder.pkl",
    "label_binarizer_path": "starter/model/label_binarizer.pkl",
    "data": [
        [
            39, "State-gov", 77516, "Bachelors", 13, "Never-married",
            "Adm-clerical", "Not-in-family", "White", "Male", 2174, 0,
            40, "United-States"
        ]
    ],
    "columns": [
        "age", "workclass", "fnlgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country"
    ]
}
url = "https://last-project.onrender.com/inference/"
response = requests.post(url=url, data=json.dumps(data))
print("Status Code: ", response.status_code)
print("Result: ", response.json()['prediction'])
