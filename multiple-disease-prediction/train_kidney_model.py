import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

df = pd.read_csv('data/kidney_disease.csv')

if 'id' in df.columns:
    df.drop('id', axis=1, inplace=True)

df.dropna(inplace=True)

le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col].astype(str))

X = df.drop('classification', axis=1)
y = df['classification']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


print("✅ Model trained successfully!")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/kidney_model.pkl')
joblib.dump((X_train, X_test, y_train, y_test), 'models/kidney_split.pkl')
print("✅ Model and data saved to /models/")

