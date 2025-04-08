# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib

# Load dataset
df = pd.read_csv("Real_estate_valuation_data_set.csv")

# Drop unnecessary columns
df.drop(columns=["No"], inplace=True)

# Define features and target
X = df.drop(columns=["Y house price of unit area"])
y = df["Y house price of unit area"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and scaler
joblib.dump(model, "real_estate_model.pkl")
joblib.dump(scaler, "real_estate_scaler.pkl")

print("✅ Model and scaler saved successfully!")
