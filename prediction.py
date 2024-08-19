import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet

def prediction(periods):
    rice2019 = pd.read_csv('C:\\Users\\yashk\\OneDrive\\Desktop\\project\\Yash-Project-GUI\\Yash-Project-GUI\\data\\Rice_2019.csv')
    rice2020 = pd.read_csv('C:\\Users\\yashk\\OneDrive\\Desktop\\project\\Yash-Project-GUI\\Yash-Project-GUI\\data\\Rice_2020.csv')
    rice2021 = pd.read_csv('C:\\Users\\yashk\\OneDrive\\Desktop\\project\\Yash-Project-GUI\\Yash-Project-GUI\\data\\Rice_2021.csv')
    rice2022 = pd.read_csv('C:\\Users\\yashk\\OneDrive\\Desktop\\project\\Yash-Project-GUI\\Yash-Project-GUI\\data\\Rice_2022.csv')
    rice2023 = pd.read_csv('C:\\Users\\yashk\\OneDrive\\Desktop\\project\\Yash-Project-GUI\\Yash-Project-GUI\\data\\Rice_2023.csv')
    rice2024 = pd.read_csv('C:\\Users\\yashk\\OneDrive\\Desktop\\project\\Yash-Project-GUI\\Yash-Project-GUI\\data\\Rice_2024 (1).csv')
    Rice = pd.concat([rice2019, rice2020, rice2021, rice2022, rice2023, rice2024])
    
    Rice['cost of cultivation'] = 0
    Rice.loc[Rice['state'] == 'Gujarat', 'cost of cultivation'] = 24513
    Rice.loc[Rice['state'] == 'Jharkhand', 'cost of cultivation'] = 19164
    Rice.loc[Rice['state'] == 'Karnataka', 'cost of cultivation'] = 24334
    Rice.loc[Rice['state'] == 'Andhra Pradesh', 'cost of cultivation'] = 28062
    Rice.loc[Rice['state'] == 'Bihar', 'cost of cultivation'] = 18727
    Rice.loc[Rice['state'] == 'Gujarat', 'cost of cultivation'] = 24513
    Rice.loc[Rice['cost of cultivation'] == 0, 'cost of cultivation'] = 34428
    
    Rice.drop('update_date', axis=1, inplace=True)
    
    West_Bengal = Rice[Rice['state'] == 'West Bengal']
    West_Bengal['arrival_date'] = pd.to_datetime(West_Bengal['arrival_date'], format='%d/%m/%Y')
    West_Bengal['date'] = West_Bengal['arrival_date'].dt.strftime('%Y-%m')
    West_Bengal['ds'] = West_Bengal.date
    West_Bengal['y'] = West_Bengal.modal_price
    
    m = Prophet(yearly_seasonality=True, seasonality_prior_scale=0.1)
    m.fit(West_Bengal)
    
    future = m.make_future_dataframe(periods=periods)
    forecast = m.predict(future)
    
    prediction_plot = m.plot_components(forecast)
    
    return prediction_plot, forecast
    