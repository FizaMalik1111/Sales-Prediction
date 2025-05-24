import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Title
st.title("Sales Prediction App")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("Train.csv")
    return df

df = load_data()

# Preprocessing
def preprocess_data(data):
    df = data.copy()

    df['Item_Weight'].fillna(df['Item_Weight'].mean(), inplace=True)
    df['Outlet_Size'].fillna(df['Outlet_Size'].mode()[0], inplace=True)
    df['Item_Visibility'].replace(0, df['Item_Visibility'].mean(), inplace=True)
    df['Item_Fat_Content'].replace({'low fat': 'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'}, inplace=True)
    df['Outlet_Years'] = 2024 - df['Outlet_Establishment_Year']

    df['New_Item_Type'] = df['Item_Identifier'].apply(lambda x: x[:2])
    df['New_Item_Type'] = df['New_Item_Type'].map({'FD': 'Food', 'NC': 'Non-Consumable', 'DR': 'Drinks'})
    df.loc[df['New_Item_Type'] == 'Non-Consumable', 'Item_Fat_Content'] = 'Non-Edible'

    # Encode categorical columns
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

    return df

df_processed = preprocess_data(df)

# Features and target
X = df_processed.drop(['Item_Outlet_Sales', 'Item_Identifier', 'Outlet_Identifier'], axis=1)
y = df_processed['Item_Outlet_Sales']

# Model training
model = RandomForestRegressor()
model.fit(X, y)

st.success("Model trained successfully!")

# Upload CSV for prediction
st.header("Upload Data for Prediction")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    input_df = pd.read_csv(uploaded_file)
    input_processed = preprocess_data(input_df)
    input_X = input_processed[X.columns]
    predictions = model.predict(input_X)

    input_df['Predicted_Sales'] = predictions
    st.write(input_df)

    csv = input_df.to_csv(index=False).encode()
    st.download_button("Download Predictions", csv, "predicted_sales.csv", "text/csv")
