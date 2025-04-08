import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load model and scaler
model = joblib.load("real_estate_model.pkl")
scaler = joblib.load("real_estate_scaler.pkl")

# -------------Page config ---------------
st.set_page_config(page_title="Real Estate App", layout="wide")
# Load external CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)



# ---------------- Sidebar ----------------
with st.sidebar:
    st.image("RealStateLogo.jpeg", width=100)  # use CSS if needed
    st.markdown('<div class="sidebar-title">Real Estate App</div>', unsafe_allow_html=True)
    menu = st.radio("📋 Menu", ["Dashboard", "Predict Price", "Report", "About"])

# ---------------- Header Bar ----------------
st.markdown("""
    <div class="custom-header">
        <h1>🏠 Real Estate Price Predictor</h1>
    </div>
""", unsafe_allow_html=True)

# ---------------- Page Logic ----------------
if menu == "Dashboard":
    # Style header
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

    st.metric("📈 Total Records", len(df))

    # Layout 2x2 with st.columns
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 💵 Price Distribution")
        fig1, ax1 = plt.subplots()
        ax1.hist(df['House Price'], bins=20, color="#0A9396")
        ax1.set_xlabel("Price per unit area")
        ax1.set_ylabel("Frequency")
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
        ax4.boxplot([df[df["Convenience Stores"] == i]["House Price"] for i in range(df["Convenience Stores"].max()+1)],
                    labels=[str(i) for i in range(df["Convenience Stores"].max()+1)])
        ax4.set_xlabel("Number of Convenience Stores")
        ax4.set_ylabel("Price per unit area")
        st.pyplot(fig4)

# ---------------- Page: Predict Price ----------------
elif menu == "Predict Price":
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

    # Two-column input layout
    col1, col2 = st.columns(2)

    with col1:
        transaction_date = st.slider("📅 Transaction Date", 2012.0, 2013.5, 2013.0, step=0.1)
        house_age = st.slider("🏚️ House Age (years)", 0, 50, 10)
        convenience = st.slider("🏪 Convenience Stores Nearby", 0, 10, 2)

    with col2:
        dist_mrt = st.number_input("🚇 Distance to Nearest MRT (meters)", min_value=0.0, max_value=10000.0, value=1000.0)
        latitude = st.number_input("🌍 Latitude", value=24.95, step=0.01)
        longitude = st.number_input("🌍 Longitude", value=121.55, step=0.01)

    # Predict button
    if st.button("📊 Predict Price"):
        input_data = np.array([[transaction_date, house_age, dist_mrt, convenience, latitude, longitude]])
        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)

        st.success(f"💰 Estimated Price: **{prediction[0]:.2f}** per unit area")

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

elif menu == "Report":
    st.subheader("📄 Report")
    st.info("This section can be expanded with saved predictions, PDF generation, or export options.")

elif menu == "About":
    st.subheader("📚 About This App")
    st.write("""
    This application was developed as part of the ICT619 Artificial Intelligence course at Murdoch University.
    It predicts real estate prices based on property features using a machine learning model.
    
    **Team:** R-Tech Solution  
    **Tech Stack:** Python, Streamlit, scikit-learn  
    """)

# ---------------- Footer ----------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<center style='color:gray;'>© 2025 R-Tech Solution | Streamlit App</center>", unsafe_allow_html=True)
