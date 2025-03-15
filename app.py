import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Title
st.title("Sales Forecasting App")

# File Upload
uploaded_file = st.file_uploader("Train.csv", type=["csv"])

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview")
    st.write(df.head())
    
    # Assume first column is date and last column is sales
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
    df = df.sort_values(by=df.columns[0])
    
    # Extract features and target
    df['Days'] = (df.iloc[:, 0] - df.iloc[:, 0].min()).dt.days
    X = df[['Days']]
    y = df.iloc[:, -1]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model training
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Display metrics
    st.write("### Model Performance")
    st.write(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.2f}")
    st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
    
    # Future predictions
    days_to_forecast = st.slider("Select days to forecast", 1, 365, 30)
    future_dates = np.array([df['Days'].max() + i for i in range(1, days_to_forecast + 1)]).reshape(-1, 1)
    future_predictions = model.predict(future_dates)
    
    # Plot results
    fig, ax = plt.subplots()
    ax.scatter(df['Days'], y, label="Actual Sales", color='blue')
    ax.plot(df['Days'], model.predict(X), label="Regression Line", color='red')
    ax.scatter(future_dates, future_predictions, label="Forecasted Sales", color='green')
    ax.set_xlabel("Days")
    ax.set_ylabel("Sales")
    ax.legend()
    st.pyplot(fig)
    
    # Display forecasted values
    forecast_df = pd.DataFrame({
        "Days from Start": future_dates.flatten(),
        "Forecasted Sales": future_predictions.flatten()
    })
    st.write("### Forecasted Sales Data")
    st.write(forecast_df)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Title
st.title("Sales Forecasting App")

# File Upload
uploaded_file = st.file_uploader("Train.csv", type=["csv"])

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview")
    st.write(df.head())
    
    # Assume first column is date and last column is sales
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
    df = df.sort_values(by=df.columns[0])
    
    # Extract features and target
    df['Days'] = (df.iloc[:, 0] - df.iloc[:, 0].min()).dt.days
    X = df[['Days']]
    y = df.iloc[:, -1]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model training
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Display metrics
    st.write("### Model Performance")
    st.write(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.2f}")
    st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
    
    # Future predictions
    days_to_forecast = st.slider("Select days to forecast", 1, 365, 30)
    future_dates = np.array([df['Days'].max() + i for i in range(1, days_to_forecast + 1)]).reshape(-1, 1)
    future_predictions = model.predict(future_dates)
    
    # Plot results
    fig, ax = plt.subplots()
    ax.scatter(df['Days'], y, label="Actual Sales", color='blue')
    ax.plot(df['Days'], model.predict(X), label="Regression Line", color='red')
    ax.scatter(future_dates, future_predictions, label="Forecasted Sales", color='green')
    ax.set_xlabel("Days")
    ax.set_ylabel("Sales")
    ax.legend()
    st.pyplot(fig)
    
    # Display forecasted values
    forecast_df = pd.DataFrame({
        "Days from Start": future_dates.flatten(),
        "Forecasted Sales": future_predictions.flatten()
    })
    st.write("### Forecasted Sales Data")
    st.write(forecast_df)
