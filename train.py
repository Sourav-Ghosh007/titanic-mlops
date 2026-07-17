# train.py
# Toy project: Titanic survival classifier
# Goal: practice the ML -> MLOps pipeline, not build a fancy model

import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# -------------------------------
# STEP 1: Load the dataset
# -------------------------------
# seaborn ships with the Titanic dataset built in - no Kaggle download needed
df = sns.load_dataset('titanic')

print("Shape of dataset:", df.shape)
print(df.head())
print(df.info())

# -------------------------------
# STEP 2: Basic EDA (Exploratory Data Analysis)
# -------------------------------
# Check for missing values - this tells us what needs cleaning
print("\nMissing values per column:")
print(df.isnull().sum())

# Check survival rate - is the data balanced or imbalanced?
print("\nSurvival counts:")
print(df['survived'].value_counts())

# -------------------------------
# STEP 3: Clean and select features
# -------------------------------
# Keep it simple - pick a few meaningful columns only
features = ['pclass', 'sex', 'age', 'fare', 'sibsp', 'parch']
target = 'survived'

data = df[features + [target]].copy()

# Fill missing age with the median (simple strategy, fine for a toy project)
data['age'] = data['age'].fillna(data['age'].median())

# Convert 'sex' (text) into numbers - models only understand numbers
data['sex'] = data['sex'].map({'male': 0, 'female': 1})

# Drop any remaining rows with missing values
data = data.dropna()

print("\nFinal dataset shape after cleaning:", data.shape)

# -------------------------------
# STEP 4: Train/test split
# -------------------------------
X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# -------------------------------
# STEP 5: Train the model
# -------------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -------------------------------
# STEP 6: Evaluate the model
# -------------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")
print("\nClassification report:")
print(classification_report(y_test, y_pred))

# -------------------------------
# STEP 7: Save the trained model as a pickle file
# -------------------------------
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\nModel saved as model.pkl")
print("This file is what your FastAPI/Streamlit app will load later.")