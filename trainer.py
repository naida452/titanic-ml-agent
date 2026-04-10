import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix
)
from xgboost import XGBClassifier

MODELS = {
    "LogisticRegression": LogisticRegression,
    "RandomForestClassifier": RandomForestClassifier,
    "XGBClassifier": XGBClassifier,
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "SVC": SVC,
}

def train_and_evaluate(df: pd.DataFrame, plan: dict) -> str:
    # Fill any remaining NA values before training
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype in ["float64", "int64"]:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])

    # Convert all columns to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model_name = plan.get("model", "LogisticRegression")
    if model_name not in MODELS:
        print(f"Model '{model_name}' not supported, defaulting to LogisticRegression.")
        model_name = "LogisticRegression"

    model_class = MODELS[model_name]
    hyperparameters = plan.get("hyperparameters", {})

    if model_name == "XGBClassifier":
        hyperparameters.setdefault("eval_metric", "logloss")
        hyperparameters.setdefault("verbosity", 0)

    model = model_class(**hyperparameters)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    output = []
    output.append(f"\nModel: {model_name}")
    output.append("\nConfusion Matrix:")
    output.append("--")
    for row in cm:
        output.append("  ".join(map(str, row)))
    output.append("--")
    output.append(f"Accuracy:  {accuracy:.2%}")
    output.append(f"F1:        {f1:.2f}")
    output.append(f"Precision: {precision:.2f}")
    output.append(f"Recall:    {recall:.2f}")

    return "\n".join(output)