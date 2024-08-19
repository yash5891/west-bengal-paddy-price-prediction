import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import pickle
import io

# Function to load data from uploaded CSV files
def load_data(uploaded_files):
    dataframes = []
    for uploaded_file in uploaded_files:
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            dataframes.append(data)
    if dataframes:
        return pd.concat(dataframes, ignore_index=True)
    else:
        return pd.DataFrame()

# Function to save objects to a pickle file
def save_pickle(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

# Function to plot forecast
def plot_forecast(m, forecast):
    fig, ax = plt.subplots(figsize=(10, 6))
    m.plot(forecast, ax=ax, xlabel='Date', ylabel='Price')
    plt.title('Forecast of Rice Prices')
    st.pyplot(fig)

# Streamlit app
st.title('west_bengal_rice_forecatsing')

# Upload CSV files
uploaded_files = st.file_uploader("Upload CSV files", type=['csv'], accept_multiple_files=True)

if uploaded_files:
    # Load and concatenate data
    Rice = load_data(uploaded_files)
    st.write("Columns in the dataset:", Rice.columns)  # Display column names for debugging

    # Ask user for column names
    date_col = st.text_input('Enter the column name for dates:', 'arrival_date')
    price_col = st.text_input('Enter the column name for prices:', 'modal_price')

    # Data manipulation
    Rice['cost of cultivation'] = 0


    
    # Filter data for West Bengal
    West_Bengal = Rice[Rice['state'] == 'West Bengal']
    
    # Convert date columns to datetime
    West_Bengal['arrival_date'] = pd.to_datetime(West_Bengal['arrival_date'], format='%d/%m/%Y')
    
    # Prepare data for Prophet
    West_Bengal = West_Bengal[['arrival_date', 'modal_price']].rename(columns={'arrival_date': 'ds', 'modal_price': 'y'})
    
    # Fit the Prophet model
    m = Prophet(yearly_seasonality=True, seasonality_prior_scale=0.1)
    m.fit(West_Bengal)
    
    # Save the model
    save_pickle(m, 'prophet_model.pkl')
    
    # Prediction
    periods = st.slider('Select number of days for prediction', min_value=30, max_value=365, value=365)
    future = m.make_future_dataframe(periods=periods, freq='D')
    forecast = m.predict(future)
    
    # Save forecast results
    save_pickle(forecast, 'forecast_results.pkl')
    
    # Plot components
    st.subheader('Forecast Components')
    fig_components = m.plot_components(forecast)
    st.pyplot(fig_components)
    
    # Plot forecast
    st.subheader('Forecast')
    plot_forecast(m, forecast)
    
    # Display forecast data
    st.subheader('Forecast Data')
    
    st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], width=1000, height=600)
  
else:
    st.write("Please upload CSV files to continue.")
