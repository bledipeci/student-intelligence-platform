import os
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import run_pipeline, get_feature_columns

# absolute path so the models folder is always found regardless of where the app is launched from
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def build_targets(df):
    # target for linear regression — predict the actual final grade as a number
    df["target_grade"] = df["G3"]

    # target for logistic regression — pass is 1 if grade is 10 or above, fail is 0
    # 10 out of 20 is the passing grade in the portuguese grading system
    df["target_pass"]  = (df["G3"] >= 10).astype(int)

    # target for random forest — group students into three performance buckets
    def categorise(grade):
        if grade >= 15:
            return "High"
        elif grade >= 10:
            return "Medium"
        else:
            return "Low"

    df["target_category"] = df["G3"].apply(categorise)
    return df


def train_and_evaluate():
    # load the feature table and add the target columns
    df = run_pipeline()
    df = build_targets(df)
    X  = df[get_feature_columns()]

    # model 1: linear regression — predicts the final grade as a continuous number
    # we split 80% for training and 20% for testing, random_state makes it reproducible
    y_grade = df["target_grade"]
    X_train, X_test, y_train, y_test = train_test_split(X, y_grade, test_size=0.2, random_state=42)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    # mae is the average difference between predicted and actual grades
    mae = mean_absolute_error(y_test, lr.predict(X_test))
    print("--- Linear Regression (Grade Prediction) ---")
    print(f"   MAE: {mae:.2f}  (out of 20 points)")

    # model 2: logistic regression — classifies whether a student will pass or fail
    y_pass = df["target_pass"]
    X_train, X_test, y_train, y_test = train_test_split(X, y_pass, test_size=0.2, random_state=42)
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)
    y_pred  = log_reg.predict(X_test)
    print("\n--- Logistic Regression (Pass/Fail) ---")
    print(f"   Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    # confusion matrix shows how many we got right and wrong for each class
    print(f"   Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    # model 3: random forest — classifies students into low, medium, or high performance
    # it trains 100 decision trees and takes a majority vote for each prediction
    y_cat = df["target_category"]
    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    # cross-validation splits the data 5 different ways and averages the accuracy
    # this gives a more reliable score than a single train/test split
    cv_scores = cross_val_score(rf, X, y_cat, cv=5)
    print("\n--- Random Forest (Performance Category) ---")
    print(f"   Accuracy:           {accuracy_score(y_test, y_pred):.2%}")
    print(f"   Cross-val (5-fold): {cv_scores.mean():.2%} (+/- {cv_scores.std():.2%})")

    # save all three models to disk so we don't retrain every time the app starts
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(lr,      os.path.join(MODELS_DIR, "grade_predictor.pkl"))
    joblib.dump(log_reg, os.path.join(MODELS_DIR, "pass_fail_classifier.pkl"))
    joblib.dump(rf,      os.path.join(MODELS_DIR, "performance_categoriser.pkl"))

    print("\nModels saved to models/")
    return lr, log_reg, rf


def load_models():
    # load the saved models from disk instead of retraining
    lr      = joblib.load(os.path.join(MODELS_DIR, "grade_predictor.pkl"))
    log_reg = joblib.load(os.path.join(MODELS_DIR, "pass_fail_classifier.pkl"))
    rf      = joblib.load(os.path.join(MODELS_DIR, "performance_categoriser.pkl"))
    return lr, log_reg, rf


if __name__ == "__main__":
    train_and_evaluate()
