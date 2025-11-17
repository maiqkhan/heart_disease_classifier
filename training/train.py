import pandas as pd 
import numpy as np
from  matplotlib import pyplot as plt
import joblib

from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, recall_score, precision_score, f1_score, make_scorer, classification_report
)

from xgboost import XGBClassifier

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def load_data() -> pd.DataFrame:
    return pd.read_csv("data/raw/heart_disease_data.csv")


def train_model(df: pd.DataFrame) -> Pipeline:
    NUMERIC_FEATURES = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    CATEGORICAL_FEATURES = ['Sex', 'ChestPainType', 'RestingECG', 'ST_Slope', 'ExerciseAngina']
    BINARY_FEATURES = ['FastingBS']

    y = df['HeartDisease'].values
    X = df.drop('HeartDisease', axis=1)

    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, random_state=RANDOM_SEED, test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, random_state=RANDOM_SEED, test_size=0.25)

    preprocessor = ColumnTransformer(
    transformers=[
        ('scaler', StandardScaler(), NUMERIC_FEATURES),
        ('onehot', OneHotEncoder(drop=None, handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ],
    remainder="passthrough"
    )

    log_pipeline = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=RANDOM_SEED, max_iter=1_000))
    ]
    )

    param_grid = {
    'classifier__penalty': ['l1', 'l2'],
    'classifier__C': [ 0.1, 1, 10],
    'classifier__solver': ['liblinear', 'saga'],
    'classifier__class_weight': ['balanced', {0: 1, 1: 1.5}, {0: 1, 1: 2}, {0: 1, 1: 2.5}],
    'classifier__max_iter': [500, 1000]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    grid_search_model = GridSearchCV(
        estimator=log_pipeline,
        param_grid=param_grid,
        scoring='recall',
        cv=cv,
        verbose=2,
        n_jobs=-1,
        return_train_score=True
    )

    grid_search_model.fit(X_train, y_train)

    best_log_pipeline = grid_search_model.best_estimator_

    y_train_pred_tuned = best_log_pipeline.predict(X_train)
    y_val_pred_tuned = best_log_pipeline.predict(X_val) 
    y_test_pred_tuned = best_log_pipeline.predict(X_test)

    log_train_recall_tuned = recall_score(y_train, y_train_pred_tuned)
    log_val_recall_tuned = recall_score(y_val, y_val_pred_tuned) 
    log_test_recall_tuned = recall_score(y_test, y_test_pred_tuned)

    print(f"recall scores:\ntrain:{log_train_recall_tuned}\nvalidation:{log_val_recall_tuned}\ntest:{log_val_recall_tuned}")

    return best_log_pipeline

def save_model(pipeline: Pipeline, output_file:str) -> None:
    with open(output_file, "wb") as f:
        joblib.dump(pipeline, f)

def main():
    df = load_data()
    model_pipeline = train_model(df)
    save_model(model_pipeline, "models/log_regression.joblib")

    print("Model saved to: models/log_regression.joblib")


if __name__ == "__main__":
    main()


        