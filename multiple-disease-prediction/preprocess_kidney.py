import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/kidney_disease.csv')
df = df.dropna()

le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col].astype(str))

X = df.drop('classification', axis=1)
y = df['classification']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

import joblib
joblib.dump((X_train, X_test, y_train, y_test), 'models/kidney_split.pkl')

print("✅ Kidney data preprocessing completed and saved.")
