from flask import Flask, jsonify
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import requests
import json

app = Flask(__name__)


def get_booking_data_from_api():
    api_url = "http://13.126.57.93:8000/admin_app/weekly_income/"  
    try:
        response = requests.get(api_url)

        if response.status_code == 200:
            data = response.json()
            messages = data['message']
            df = pd.DataFrame(messages)
            df['end_date'] = pd.to_datetime(df['end_date'])

            df.sort_values(by=['turf__id', 'end_date'], inplace=True)

            df['booking_LastWeek'] = df.groupby('turf__id')['total_booking'].shift(1)
            df['booking_2Weeksback'] = df.groupby('turf__id')['total_booking'].shift(2)
            df['booking_3Weeksback'] = df.groupby('turf__id')['total_booking'].shift(3)

            df.fillna(0, inplace=True)

            book_df = df[['turf__id', 'end_date', 'booking_LastWeek', 'booking_2Weeksback', 'booking_3Weeksback', 'total_booking']]

            return book_df
        else:
            print(f"API request failed with status code: {response.status_code}")

    except requests.exceptions.RequestException as e:
        print(f"Error making API request: {e}")

    return None

def train_models(data):
    models = {}
    
    for turf_id in data['turf__id'].unique():
        turf_data = data[data['turf__id'] == turf_id].copy()
        
        X_turf = turf_data[['booking_LastWeek', 'booking_2Weeksback', 'booking_3Weeksback']]
        y_turf = turf_data['total_booking']
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)  
        model.fit(X_turf, y_turf)
        
        models[turf_id] = model
    
    return models




@app.route('/booking', methods=['GET', 'POST'])
def booking():
    book_df = get_booking_data_from_api()

    if book_df is not None:
        models = train_models(book_df)

        predictions = []

        for turf_id in book_df['turf__id'].unique():
            turf_data = book_df[book_df['turf__id'] == turf_id].copy()

            if not turf_data.empty:
                recent_row = turf_data.iloc[-1]

                try:
                    input_features = [
                        recent_row['total_booking'],
                        recent_row['booking_LastWeek'],
                        recent_row['booking_2Weeksback']
                    ]

                    model_to_use = models.get(turf_id)
                    if model_to_use is not None:
                        predicted_booking = model_to_use.predict([input_features])[0]
                        rounded_predicted_booking = round(predicted_booking)
                    else:
                        rounded_predicted_booking = 2
                except Exception as e:
                    print(f"There was an error during income prediction for turf_id {turf_id}: {str(e)}")
                    rounded_predicted_booking = 2

                predictions.append({"turf__id": turf_id, "predicted_booking": rounded_predicted_booking})
            else:
                predictions.append({"turf__id": turf_id, "error": "No data found for the specified turf_id"})

        predictions_df = pd.DataFrame(predictions)

        return predictions_df.to_json(orient='records')
    else:
        print("Error fetching data from the API")
        return jsonify({"error": "Error fetching data from the API"})


if __name__ == '__main__':
    app.run(debug=True)