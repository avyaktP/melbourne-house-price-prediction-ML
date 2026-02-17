import joblib
import os
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from data_cleaning import load_data, clean_data
from feature_engineering import prepare_features
from sklearn.model_selection import cross_val_score



def train_and_compare(path):

    df = clean_data(load_data(path))

    X_train, X_test, y_train, y_test = prepare_features(df)

    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(),
        "DecisionTree": DecisionTreeRegressor(),
        "RandomForest": RandomForestRegressor(
            n_estimators=400,
            max_depth=25,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            random_state=42
        )
    }

    results = {}

    for name, model in models.items():

        model.fit(X_train, y_train)

        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=5,
            scoring="r2"
        )

        score = cv_scores.mean()


        results[name] = round(score, 3)

    # saving the best model
    best_model = models["RandomForest"]

    os.makedirs("../models", exist_ok=True)

    joblib.dump(
        best_model,
        "../models/random_forest_model.pkl"
    )


    return results


if __name__ == "__main__":

    data_path = "../data/Melbourne_housing_FULL.csv"

    scores = train_and_compare(data_path)

    print("\nModel Performance (R² Scores):\n")

    for model, score in scores.items():
        print(f"{model}: {score}")