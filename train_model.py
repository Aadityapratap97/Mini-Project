import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("burnout_dataset.csv")

# Label Encoding
encoders = {}

categorical_columns = [
    "Stress_Level",
    "Physical_Activity",
    "Assignment_Load",
    "Gender",
    "Burnout"
]

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Features and Target
X = df.drop("Burnout", axis=1)
y = df["Burnout"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("=" * 50)
print(f"Model Accuracy : {accuracy*100:.2f}%")
print("=" * 50)
print("\nClassification Report\n")
print(classification_report(y_test, y_pred))

# Save Model
joblib.dump(model, "model.pkl")
joblib.dump(encoders, "encoders.pkl")

print("\nModel saved successfully as model.pkl")
print("Encoders saved successfully as encoders.pkl")