import joblib
import pandas as pd


MODEL_PATH = "../models/random_forest_model.pkl"


def predict_price(input_data):

    model = joblib.load(MODEL_PATH)

    df = pd.DataFrame([input_data])

    prediction = model.predict(df)[0]

    return prediction


if __name__ == "__main__":

    sample_house = {
        "Rooms": 3,
        "Bathroom": 2,
        "Car": 1,
        "Distance": 10,
        "Landsize": 300,
        "BuildingArea": 150,
        "YearBuilt": 2005
    }

    price = predict_price(sample_house)

    print(f"Predicted Price: ${int(price)}")