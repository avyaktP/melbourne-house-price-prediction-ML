import pandas as pd
from sklearn.model_selection import train_test_split


def prepare_features(df):

    categorical = ["Suburb", "Type"]

    df = pd.get_dummies(
        df,
        columns=categorical,
        drop_first=True
    )

    X = df.drop("Price", axis=1)
    y = df["Price"]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )