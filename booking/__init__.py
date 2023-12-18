from flask import Flask, jsonify
import joblib
import pandas as pd
import numpy as np
import requests
import json

app = Flask(__name__)

pipeline = joblib.load('Future_Bookings_model.joblib')

def get_booking_data_from_api():
    api_url = "https://8990-116-68-110-250.ngrok-free.app/admin_app/weekly_income/"  
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

            df.fillna(2, inplace=True)

            book_df = df[['turf__id', 'end_date', 'booking_LastWeek', 'booking_2Weeksback', 'booking_3Weeksback', 'total_booking']]

            return book_df
        else:
            print(f"API request failed with status code: {response.status_code}")

    except requests.exceptions.RequestException as e:
        print(f"Error making API request: {e}")

    # Return default values in case of an exception
    return None

@app.route('/booking', methods=['GET', 'POST'])
def booking():
    book_df = get_booking_data_from_api()

    if book_df is not None:
        predictions = []

        for turf__id in book_df['turf__id'].unique():
            turf_data = book_df[book_df['turf__id'] == turf__id]

            if not turf_data.empty:
                recent_row = turf_data.iloc[-1]  # Get the last row for the current turf__id

                try:
                    input_features = [
                        recent_row['booking_LastWeek'],
                        recent_row['booking_2Weeksback'],
                        recent_row['booking_3Weeksback']
                    ]

                    predicted_booking = pipeline.predict([input_features])[0]
                    rounded_predicted_booking = round(predicted_booking)

                except Exception as e:
                    print(f"There was an error during booking prediction for turf__id {turf__id}: {e}")
                    # Set a default value for prediction
                    rounded_predicted_booking = 50  # Set your desired default value

                predictions.append({"turf_id": turf__id, "predicted_bookings": rounded_predicted_booking})
            else:
                predictions.append({"turf_id": turf__id, "error": "No data found for the specified turf_id"})

        predictions_df = pd.DataFrame(predictions)

        return predictions_df.to_json(orient='records')
    else:
        print("Error fetching data from the API")
        return jsonify({"error": "Error fetching data from the API"})

if __name__ == '__main__':
    app.run(debug=True)