import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Preprocessing
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Remove customerID
df = df.drop("customerID", axis=1)

# Encode categorical columns
df_encoded = pd.get_dummies(df, drop_first=True)

# Features and target
X = df_encoded.drop("Churn", axis=1)
y = df_encoded["Churn"]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = XGBClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Save model and feature columns
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/churn_model.pkl")
joblib.dump(X_train.columns.tolist(), "models/feature_columns.pkl")

print("Model saved successfully in models/churn_model.pkl")
print("Feature columns saved successfully")