# 🏠 Real Estate Price Predictor

A user-friendly and interactive web application built with **Streamlit** that predicts real estate prices based on key features such as location, age, and accessibility. This project was developed as part of the ICT619 Artificial Intelligence course at **Murdoch University**.

Access App: https://realestate-valuation-prediction.streamlit.app
---

## 🚀 Features

### ✅ Prediction Engine
- Predicts real estate unit prices using a trained **Random Forest** regression model.
- Scales input features for accurate ML results using **StandardScaler**.
- Visualizes feature importance to understand what influences price most.

### 📊 Dashboard Visualizations
- Shows total number of records in the dataset.
- Interactive 2x2 grid charts:
  - 💵 Price distribution
  - 🏚️ Price vs House Age
  - 🚇 Price vs Distance to MRT
  - 🏪 Price vs Convenience Stores
- 📈 Price trend by House Age.

### 🌍 Investment Map (Interactive)
- Visualizes **all 414 properties** using `pydeck`.
- 🔵 All properties in blue.
- 🟢 Top 5 investment picks highlighted in green.
- 🔴 Highest-valued property marked in red.
- Custom tooltips show Price, Age, MRT distance, and Date.
- Zoom and center optimized for clarity.

### 🏆 Top 5 Investment Properties Table
- Calculated using a **Value Index** (Distance to MRT ÷ Price).
- Includes human-readable location names using **reverse geocoding**.
- Proper units: `NT$/Ping`, `meters`, `years`.

### 📌 Summary Metrics
- 📊 Average Price (NT$/Ping)
- 🏘️ Oldest Property Age
- 🚇 Closest MRT Distance

---

## 📂 Project Structure
Realestate-Valuation-Project/
│
├── real_estate_app.py           # Main Streamlit app
├── train_model.py               # Model training script
├── requirements.txt             # Python dependencies
├── style.css                    # Custom CSS styling
├── RealStateLogo.jpeg           # App logo
├── Real_estate_valuation_data_set.csv  # Dataset
├── real_estate_model.pkl        # Trained model
├── real_estate_scaler.pkl       # Scaler used for prediction
└── README.md                    # You're here!

## 🧪 Dataset & Feature Explanation

The dataset is sourced from a real estate valuation dataset from Taiwan. Each row represents a property sale record with the following features:

| Feature                     | Description                                             |
|----------------------------|---------------------------------------------------------|
| `Transaction Date`         | Sale year (e.g., 2012.917 = Nov 2012)                   |
| `House Age`                | Age of the house in years                               |
| `Distance to MRT`          | Distance (meters) to the nearest Metro Rail Transit     |
| `Convenience Stores`       | Number of convenience stores nearby                     |
| `Latitude` / `Longitude`   | Geographic location of the property                     |
| `House Price`              | Price per unit area (NT$/Ping - 1 Ping = 3.3 m²)        |

---

## ☁️ Deployment Guide

### 📦 Local Deployment

1. Clone the repo:
   ```bash
   git clone https://github.com/fasrinaleem/Realestate-Valuation-Project.git
   cd Realestate-Valuation-Project

2. Install dependencies:
   pip install -r requirements.txt

3. Run the app:
   streamlit run real_estate_app.py


🧪 Tech Stack
        Python
        Streamlit
        Scikit-learn
        PyDeck (Deck.gl)
        Geopy (Nominatim API)
        Pandas, NumPy, Matplotlib

👤 Authors

Developed by Fasrin Aleem, Rabinra Mahato & Kushi
Course: ICT619 Artificial Intelligence
Institution: Murdoch University
Year: 2025