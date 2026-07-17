# train_mlflow.py
# Same Titanic training pipeline, now with MLflow experiment tracking

import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("sqlite:///mlflow.db")

# -------------------------------
# Tell MLflow which experiment this run belongs to
# If it doesn't exist yet, MLflow creates it automatically
# -------------------------------
mlflow.set_experiment("Titanic Survival Prediction")

# -------------------------------
# Load and clean data (same as before)
# -------------------------------
df = sns.load_dataset('titanic')
features = ['pclass', 'sex', 'age', 'fare', 'sibsp', 'parch']
target = 'survived'

data = df[features + [target]].copy()
data['age'] = data['age'].fillna(data['age'].median())
data['sex'] = data['sex'].map({'male': 0, 'female': 1})
data = data.dropna()

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Try a few different hyperparameter combinations
# Each one becomes a separate "run" inside MLflow -
# this is how you compare experiments instead of overwriting results
# -------------------------------
hyperparameter_sets = [
    {"n_estimators": 50, "max_depth": 3},
    {"n_estimators": 100, "max_depth": 5},
    {"n_estimators": 200, "max_depth": 10},
]

for params in hyperparameter_sets:

    # start_run() = begin tracking ONE experiment attempt
    with mlflow.start_run():

        # STEP 1: Log the hyperparameters used for this run
        mlflow.log_param("n_estimators", params["n_estimators"])
        mlflow.log_param("max_depth", params["max_depth"])

        # STEP 2: Train the model with these hyperparameters
        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=42
        )
        model.fit(X_train, y_train)

        # STEP 3: Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # STEP 4: Log metrics - this is what lets you COMPARE runs later
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # STEP 5: Log the actual trained model as an artifact
        mlflow.sklearn.log_model(model, "model")

        print(f"Params: {params} -> Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

print("\nAll runs logged. Run 'mlflow ui' to see them visually.")
