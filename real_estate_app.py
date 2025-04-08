import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# -------------------- Load Model & Scaler --------------------
model = joblib.load("real_estate_model.pkl")
scaler = joblib.load("real_estate_scaler.pkl")

# -------------------- Streamlit Page Configuration --------------------
st.set_page_config(page_title="Real Estate App", layout="wide")

# Load external CSS styles
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# -------------------- Sidebar Navigation --------------------
with st.sidebar:
    st.image("RealStateLogo.jpeg", width=100)
    st.markdown('<div class="sidebar-title">Real Estate App</div>', unsafe_allow_html=True)
    menu = st.radio("📋 Menu", ["Dashboard", "Predict Price", "Report", "About"])

# -------------------- Header --------------------
st.markdown("""
    <div class="custom-header">
        <h1>🏠 Real Estate Price Predictor</h1>
    </div>
""", unsafe_allow_html=True)

# -------------------- Page: Dashboard --------------------
if menu == "Dashboard":

    # Load dataset
    df = pd.read_csv("Real_estate_valuation_data_set.csv")
    df.rename(columns={
        "X1 transaction date": "Transaction Date",
        "X2 house age": "House Age",
        "X3 distance to the nearest MRT station": "Distance to MRT",
        "X4 number of convenience stores": "Convenience Stores",
        "X5 latitude": "Latitude",
        "X6 longitude": "Longitude",
        "Y house price of unit area": "House Price"
    }, inplace=True)

    # Dashboard heading and description
    st.markdown("""
        <style>
            .dashboard-title {
                font-size: 28px;
                font-weight: bold;
                color: #0A9396;
                margin-bottom: 10px;
            }
            .dashboard-subtext {
                font-size: 16px;
                color: #666;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="dashboard-title">📊 Dashboard Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="dashboard-subtext">Insightful visualizations based on the Real Estate dataset.</div>', unsafe_allow_html=True)
    st.markdown("---")

    # Show record count
    st.markdown("""
        <div style="background-color: #1a1a1a; padding: 20px; border-radius: 10px; text-align: center;">
            <h5 style="color: white;">📈 Total Records</h5>
            <h2 style="color: #F4A261;">{}</h2>
        </div>
    """.format(len(df)), unsafe_allow_html=True)

    # 📍 Property Locations Map (Simple Version with st.map)
    st.markdown("#### 🗺️ Property Locations on Map")
    st.map(df.rename(columns={"Latitude": "latitude", "Longitude": "longitude"}))


    # Charts layout - 2x2 grid
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 💵 Price Distribution")
        fig1, ax1 = plt.subplots()
        ax1.hist(df['House Price'], bins=20, color="#0A9396", edgecolor='white')
        ax1.set_xlabel("Price per unit area")
        ax1.set_ylabel("Frequency")
        ax1.set_title("House Price Distribution", fontsize=14)
        ax1.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig1)

    with col2:
        st.markdown("#### 🏚️ Price vs House Age")
        fig2, ax2 = plt.subplots()
        ax2.scatter(df['House Age'], df['House Price'], alpha=0.6, color="#bb3e03")
        ax2.set_xlabel("House Age (years)")
        ax2.set_ylabel("Price per unit area")
        st.pyplot(fig2)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("#### 🚇 Price vs Distance to MRT")
        fig3, ax3 = plt.subplots()
        ax3.scatter(df['Distance to MRT'], df['House Price'], alpha=0.6, color="#005f73")
        ax3.set_xlabel("Distance to MRT (meters)")
        ax3.set_ylabel("Price per unit area")
        st.pyplot(fig3)

    with col4:
        st.markdown("#### 🏪 Price vs Convenience Stores Nearby")
        fig4, ax4 = plt.subplots()
        ax4.boxplot(
            [df[df["Convenience Stores"] == i]["House Price"] for i in range(df["Convenience Stores"].max() + 1)],
            labels=[str(i) for i in range(df["Convenience Stores"].max() + 1)]
        )
        ax4.set_xlabel("Number of Convenience Stores")
        ax4.set_ylabel("Price per unit area")
        st.pyplot(fig4)

# -------------------- Page: Predict Price --------------------
elif menu == "Predict Price":
    # Styling for prediction card
    st.markdown("""
        <style>
            .predict-card {
                background-color: #f9f9f9;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
            }
            .predict-title {
                font-size: 26px;
                font-weight: bold;
                color: #333;
                margin-bottom: 20px;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="predict-title">🔍 Predict Property Value</div>', unsafe_allow_html=True)
    st.markdown('<div class="predict-card">', unsafe_allow_html=True)

    # User inputs
    col1, col2 = st.columns(2)
    with col1:
        transaction_date = st.slider("📅 Transaction Date", 2012.0, 2013.5, 2013.0, step=0.1)
        house_age = st.slider("🏚️ House Age (years)", 0, 50, 10)
        convenience = st.slider("🏪 Convenience Stores Nearby", 0, 10, 2)
    with col2:
        dist_mrt = st.number_input("🚇 Distance to Nearest MRT (meters)", min_value=0.0, max_value=10000.0, value=1000.0)
        latitude = st.number_input("🌍 Latitude", value=24.95, step=0.01, help="Enter between 24.9 and 25.1")
        longitude = st.number_input("🌍 Longitude", value=121.55, step=0.01)

    # Prediction logic
    if st.button("📊 Predict Price"):
        input_data = np.array([[transaction_date, house_age, dist_mrt, convenience, latitude, longitude]])
        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)

        st.success(f"💰 Estimated Price: **{prediction[0]:.2f}** per unit area")

        # Feature importance chart
        st.markdown("### 🔎 Feature Importance")
        features = ['Transaction Date', 'House Age', 'Distance to MRT', 'Convenience Stores', 'Latitude', 'Longitude']
        importances = model.feature_importances_
        importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance')

        fig, ax = plt.subplots()
        ax.barh(importance_df['Feature'], importance_df['Importance'], color='#0A9396')
        ax.set_xlabel("Importance")
        ax.set_title("Top Predictive Features")
        st.pyplot(fig)

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------- Page: Report --------------------
elif menu == "Report":
    st.subheader("📄 Report")
    st.info("This section can be expanded with saved predictions, PDF generation, or export options.")

# -------------------- Page: About --------------------
elif menu == "About":
    st.subheader("📚 About This App")
    st.write("""
    This application was developed as part of the ICT619 Artificial Intelligence course at Murdoch University.
    It predicts real estate prices based on property features using a machine learning model.

    **Team:** R-Tech Solution  
    **Tech Stack:** Python, Streamlit, scikit-learn  
    """)

# -------------------- Footer --------------------
st.markdown("""
<hr>
<div style='text-align:center; color:gray;'>
    © 2025 R-Tech Solution | <a href='https://github.com/fasrinaleem/Realestate-Valuation-Project/tree/master' target='_blank'>Github</a>
</div>
""", unsafe_allow_html=True)
