import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_concrete_data(module, data_path=None):
    """Load the five-column concrete dataset and apply the module's scalers."""
    data_path = data_path or os.environ.get("CONCRETE_DATA", "data.csv")
    data = pd.read_csv(
        data_path,
        header=None,
        names=["x1", "x2", "x3", "x4", "y"],
        usecols=[0, 1, 2, 3, 4],
    )

    X = data[["x1", "x2", "x3", "x4"]].values
    y = data[["y"]].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train = module.standard_scaler(
        np.asarray(X_train), module.FEATURE_SCALER_PATH, mode="train"
    ).tolist()
    X_test = module.standard_scaler(
        np.asarray(X_test), module.FEATURE_SCALER_PATH, mode="val"
    ).tolist()
    y_train = module.standard_scaler(
        np.asarray(y_train), module.TARGET_SCALER_PATH, mode="train"
    ).tolist()
    y_test = module.standard_scaler(
        np.asarray(y_test), module.TARGET_SCALER_PATH, mode="val"
    ).tolist()

    return X_train, X_test, y_train, y_test
