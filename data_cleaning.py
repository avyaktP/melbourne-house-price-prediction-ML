import pandas as pd


def load_data(path):
    return pd.read_csv(path)


def clean_data(df):

    df = df.dropna(subset=["Price"])

    df = df[
        [
            "Suburb",
            "Rooms",
            "Bathroom",
            "Car",
            "Distance",
            "Landsize",
            "BuildingArea",
            "YearBuilt",
            "Lattitude",
            "Longtitude",
            "Type",
            "Price"
        ]
    ]

    df = df.dropna()

    return df