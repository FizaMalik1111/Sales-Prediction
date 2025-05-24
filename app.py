import streamlit as st
import pandas as pd
import joblib
from io import BytesIO

# Load model
@st.cache_resource
def load_model():
    return joblib.load("d02c44f1-26e0-4945-9f3e-6d8f259f331c.pkl")

model = load_model()

# Set Streamlit page config
st.set_page_config(page_title="Sales Predictor", page_icon="📈", layout="centered")

# App title and instructions
st.title("📈 Smart Sales Prediction App")
st.markdown("Upload your CSV file and get real-time sales predictions!")

# Upload CSV
uploaded_file = st.file_uploader("📤 Upload CSV", type=["csv"])

# Process uploaded CSV
if uploaded_file is not None:
    try:
        # Read data
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")

        # Show uploaded data preview
        st.subheader("🔍 Uploaded Data Preview")
        st.dataframe(df.head())

        # Generate predictions
        predictions = model.predict(df)
        df['Predicted_Sales'] = predictions

        # Show predictions
        st.subheader("📊 Predicted Sales")
        st.dataframe(df[['Predicted_Sales']].head())

        # Prepare CSV for download
        csv_out = BytesIO()
        df.to_csv(csv_out, index=False)
        st.download_button(
            label="📥 Download Predictions CSV",
            data=csv_out.getvalue(),
            file_name="predicted_sales.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"❌ An error occurred: {e}")
else:
    st.info("Please upload a CSV file to generate predictions.")

# Footer
st.markdown("---")
st.caption("🔧 Powered by Streamlit & Scikit-learn")
