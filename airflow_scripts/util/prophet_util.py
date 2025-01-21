import pandas as pd
from prophet import Prophet
import pickle
import os
import util.constant
import requests
import json

def train(df):
    '''
    Train the model according to the input data for single factory.

    Parameters:
        df - cleansed data in DataFrame format

    Output:
        model file - name with factory id

    Returns:
        prophet model object
    '''

    cur_fatory_id = df[util.constant.STATION_ID].iloc[0]
    df = df.drop(columns=[util.constant.STATION_ID])
    df = df.rename(columns={util.constant.DATETIME: 'ds', util.constant.POWER: 'y'})
    df['ds'] = pd.to_datetime(df['ds'])

    model = Prophet()
    model.fit(df)

    if not os.path.exists('saved_model'):
        os.makedirs('saved_model')
    
    with open('saved_model/prophet_model_' + str(cur_fatory_id) + '.pkl', 'wb') as f:
        pickle.dump(model, f)

    return model

def predict_all(factory_id, model):
    '''
    Predict for all the history timestamps for single factory.

    Parameters:
        factory_id - current factory id

    Output:
        prediction csv file
    '''
    
    if model is None:
        with open('saved_model/prophet_model_' + str(factory_id) + '.pkl', 'rb') as f:
            model = pickle.load(f)

    future = model.make_future_dataframe(periods=0, freq='h')
    forecast = model.predict(future)

    forecast[util.constant.STATION_ID] = factory_id
    forecast = forecast.rename(columns={'ds': 'datetime', 'yhat': 'power', util.constant.STATION_ID: 'stationId'})
    forecast = forecast[['datetime', 'power', 'stationId']]

    norm_model_file = 'saved_norm/' + str(factory_id) + '.pickle'
    with open(norm_model_file, 'rb') as f:
        saved_model = pickle.load(f)
    
    # Extract and reshape power data for inverse transform
    power_data = forecast['power'].values.reshape(-1, 1)
    
    # Inverse transform only power data
    unnormalized_power = saved_model.inverse_transform(power_data)
    
    # Create DataFrame with original datetime and denormalized power
    unnormalized_data_df = pd.DataFrame({
        'datetime': forecast['datetime'],
        'power': unnormalized_power.flatten(),
        'stationId': forecast['stationId']
    })
    
    # Convert datetime to proper format
    unnormalized_data_df['datetime'] = pd.to_datetime(unnormalized_data_df['datetime'])

    saved_path = 'predictions/'
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    filename = 'prophet_forecast_result_' + str(factory_id) + '.csv'
    unnormalized_data_df.to_csv(saved_path + filename, index=False)

    unnormalized_data_df['datetime'] = unnormalized_data_df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')

    records = unnormalized_data_df.to_dict('records')
    json_str = {'datas': records}

    try:
        response = requests.post(util.constant.database_send_api_url, json=json_str)
        response.raise_for_status()  # Raise an error for bad status codes

    except requests.exceptions.RequestException as e:
        util.constant.logger.error(f'Failed to send data for factory {factory_id}: {str(e)}')

    return

def predict_one_day(factory_id, model):
    if model is None:
        with open('saved_model/prophet_model_' + str(factory_id) + '.pkl', 'rb') as f:
            model = pickle.load(f)

    start_date = pd.Timestamp.now().normalize()  # 今天的零点
    future_dates = pd.date_range(start=start_date, periods=24*2, freq='h')
    future = pd.DataFrame({'ds': future_dates})
    forecast = model.predict(future)

    forecast[util.constant.STATION_ID] = factory_id
    forecast = forecast.rename(columns={'ds': 'datetime', 'yhat': 'power', util.constant.STATION_ID: 'stationId'})
    forecast = forecast[['datetime', 'power', 'stationId']]

    norm_model_file = 'saved_norm/' + str(factory_id) + '.pickle'
    with open(norm_model_file, 'rb') as f:
        saved_model = pickle.load(f)
    
    # Extract and reshape power data for inverse transform
    power_data = forecast['power'].values.reshape(-1, 1)
    
    # Inverse transform only power data
    unnormalized_power = saved_model.inverse_transform(power_data)
    
    # Create DataFrame with original datetime and denormalized power
    unnormalized_data_df = pd.DataFrame({
        'datetime': forecast['datetime'],
        'power': unnormalized_power.flatten(),
        'stationId': forecast['stationId']
    })
    
    # Convert datetime to proper format
    unnormalized_data_df['datetime'] = pd.to_datetime(unnormalized_data_df['datetime'])

    saved_path = 'predictions/'
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    filename = 'prophet_forecast_result_' + str(factory_id) + '.csv'
    unnormalized_data_df.to_csv(saved_path + filename, index=False)

    unnormalized_data_df['datetime'] = unnormalized_data_df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')

    records = unnormalized_data_df.to_dict('records')
    json_str = {'datas': records}

    try:
        response = requests.post(util.constant.database_send_api_url, json=json_str)
        response.raise_for_status()  # Raise an error for bad status codes

    except requests.exceptions.RequestException as e:
        util.constant.logger.error(f'Failed to send data for factory {factory_id}: {str(e)}')

    return