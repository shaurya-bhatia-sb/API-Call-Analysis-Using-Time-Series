from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import os

app = Flask(__name__)

# Load main dataset
file_path = 'API Call Dataset.csv'
data = pd.read_csv(file_path)

# Helper function to get top N most frequently called APIs
def get_top_apis(n=0):
    return data['API Code'].value_counts().head(n).index.tolist()

# Function to evaluate models for a selected API
def evaluate_models(api_code):
    # Load specific API data
    api_data = data[data['API Code'] == api_code]
    api_data['Time of Call'] = pd.to_datetime(api_data['Time of Call'], errors='coerce')
    api_data.dropna(subset=['Time of Call'], inplace=True)
    api_data.set_index('Time of Call', inplace=True)

    # Resample data
    daily_counts = api_data.resample('D').size()
    print(f"API: {api_code}")
    print(daily_counts.describe())  # Debugging output

    # Split into train and test
    split_index = int(len(daily_counts) * 0.8)
    train, test = daily_counts[:split_index], daily_counts[split_index:]

    performance = {}
    forecasts = {'Dates': test.index}

    try:
        model_ses = ExponentialSmoothing(train).fit()
        forecasts['SES'] = model_ses.forecast(len(test))
        performance['SES'] = mean_squared_error(test, forecasts['SES'])
    except Exception as e:
        print(f"SES error: {e}")
        forecasts['SES'] = [np.nan] * len(test)
        performance['SES'] = np.inf

    try:
        model_hw = ExponentialSmoothing(
            train, trend='add', seasonal='add', seasonal_periods=7
        ).fit()
        forecasts['Holt-Winters'] = model_hw.forecast(len(test))
        performance['Holt-Winters'] = mean_squared_error(test, forecasts['Holt-Winters'])
    except Exception as e:
        print(f"Holt-Winters error: {e}")
        forecasts['Holt-Winters'] = [np.nan] * len(test)
        performance['Holt-Winters'] = np.inf

    try:
        model_arima = ARIMA(train, order=(5, 1, 0)).fit()
        forecasts['ARIMA'] = model_arima.forecast(len(test))
        performance['ARIMA'] = mean_squared_error(test, forecasts['ARIMA'])
    except Exception as e:
        print(f"ARIMA error: {e}")
        forecasts['ARIMA'] = [np.nan] * len(test)
        performance['ARIMA'] = np.inf

    best_model = min(performance, key=performance.get)
    return best_model, performance, forecasts


# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/set_top_n', methods=['POST'])
def set_top_n():
    top_n = int(request.form['top_n'])
    top_apis = get_top_apis(top_n)
    return render_template('top_apis.html', top_apis=top_apis)

@app.route('/select_api', methods=['POST'])
def select_api():
    api_code = request.form['api_code']
    return render_template('api_options.html', api_code=api_code)

@app.route('/view_api_calls', methods=['POST'])
def view_api_calls():
    api_code = request.form['api_code']
    api_data = data[data['API Code'] == api_code]
    api_data['Time of Call'] = pd.to_datetime(api_data['Time of Call'], errors='coerce')
    api_data.dropna(subset=['Time of Call'], inplace=True)
    api_data.sort_values(by='Time of Call', inplace=True)
    api_calls_list = api_data['Time of Call'].tolist()
    return render_template('view_calls.html', api_code=api_code, api_calls=api_calls_list)

@app.route('/download_api_calls', methods=['POST'])
def download_api_calls():
    api_code = request.form['api_code']
    api_data = data[data['API Code'] == api_code]
    file_name = f'{api_code}_calls.csv'
    api_data.to_csv(file_name, index=False)
    return send_file(file_name, as_attachment=True)
@app.route('/show_performance', methods=['POST'])
def show_performance():
    api_code = request.form['api_code']
    selected_model = request.form.get('model', 'best')  # Default to 'best' if no model is selected

    # Evaluate models
    best_model, performance, forecasts = evaluate_models(api_code)

    # Choose the forecast based on selected model
    if selected_model == 'SES':
        model_forecast = forecasts['SES']
    elif selected_model == 'Holt-Winters':
        model_forecast = forecasts['Holt-Winters']
    elif selected_model == 'ARIMA':
        model_forecast = forecasts['ARIMA']
    else:
        model_forecast = forecasts[best_model]  # Default to the best model

    # Prepare only dates for display
    forecast_dates = [str(date) for date in forecasts['Dates']]  # Convert dates to strings

    return render_template(
        'performance.html',
        api_code=api_code,
        best_model=best_model,
        performance=performance,
        forecast_dates=forecast_dates,
        available_models=['SES', 'Holt-Winters', 'ARIMA'],  # Available options
        selected_model=selected_model
    )


if __name__ == '__main__':
    app.run(debug=True)
