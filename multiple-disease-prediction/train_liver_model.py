import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

df = pd.read_csv("data/indian_liver_patient.csv")

df['Gender'] = LabelEncoder().fit_transform(df['Gender'])


df['Albumin_and_Globulin_Ratio'].fillna(df['Albumin_and_Globulin_Ratio'].mean(), inplace=True)


df['Dataset'] = df['Dataset'].apply(lambda x: 1 if x == 1 else 0)

X = df.drop('Dataset', axis=1)
y = df['Dataset']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Liver Disease Prediction Accuracy:", accuracy_score(y_test, y_pred))
print("\nReport:\n", classification_report(y_test, y_pred))

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/liver_model.pkl")
