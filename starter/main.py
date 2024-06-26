# Put the code for your API here.

from fastapi import FastAPI
from typing import Union
from pydantic import BaseModel
import numpy as np
import pandas as pd
import pickle
import logging

from pathlib import Path
import sys
import os
cwd = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(cwd, 'starter'))

from ml.data import process_data  # type: ignore # noqa
from ml.model import train_model, inference, compute_model_metrics  # type: ignore # noqa

logger = logging.getLogger('uvicorn.error')
logger.setLevel(logging.DEBUG)
# Instantiate the app.
app = FastAPI()

# Define a GET on the specified endpoint.


class Prediction(BaseModel):
    path_model: str
    encoder_path: str
    label_binarizer_path: str
    data: list
    columns: list

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "path_model": "model/clf_model.pkl",
                    "encoder_path": "model/encoder.pkl",
                    "label_binarizer_path": "model/label_binarizer.pkl",
                    "data": [[39, "State-gov", 77516, "Bachelors", 13, "Never-married", "Adm-clerical", "Not-in-family",
                             "White", "Male", 2174, 0, 40, "United-States"]],
                    "columns": ["age", "workclass", "fnlgt", "education", "education-num", "marital-status", "occupation",
                                "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country"]
                }
            ]
        }
    }


@app.get("/")
async def say_hello():
    logger.info("INFO")
    return {"greeting": "Welcome to the last udacity MLDevOps Training project!"}


@app.post("/inference/")
async def inference_api(input: Prediction):
    logger.info(input)
    df_data = pd.DataFrame.from_records(input.data)
    df_data = df_data.set_axis(input.columns, axis=1)

    # Get model
    model = pickle.load(open(input.path_model, "rb"))
    # Get encoder
    encoder = pickle.load(open(input.encoder_path, "rb"))
    # Get labeller
    label_binarizer = pickle.load(open(input.label_binarizer_path, "rb"))

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

    X, _, _, _ = process_data(
        df_data, categorical_features=cat_features, label=None, training=False, encoder=encoder, lb=label_binarizer
    )

    prediction = inference(model, X)

    return {"prediction": prediction.item()}
