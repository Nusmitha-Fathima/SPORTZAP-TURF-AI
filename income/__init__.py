from flask import Flask, jsonify
import joblib
import pandas as pd
import numpy as np
import requests
import json

app = Flask(__name__)
linear_model = joblib.load('Future_Earnings_model.joblib')

def get_data_from_api():
    api_url = "https://8990-116-68-110-250.ngrok-free.app/admin_app/weekly_income/"
    try:
        response = requests.get(api_url)

        if response.status_code == 200:
            data = response.json()
            messages = data['message']
            df = pd.DataFrame(messages)
            df['end_date'] = pd.to_datetime(df['end_date'])

            df.sort_values(by=['turf__id', 'end_date'], inplace=True)

            df['income_LastWeek'] = df.groupby('turf__id')['total_income'].shift(1)
            df['income_2Weeksback'] = df.groupby('turf__id')['total_income'].shift(2)
            df['income_3Weeksback'] = df.groupby('turf__id')['total_income'].shift(3)

            df.fillna(950, inplace=True)

            result_df = df[['turf__id', 'end_date', 'income_LastWeek', 'income_2Weeksback', 'income_3Weeksback', 'total_income']]

            return result_df
        else:
            print(f"API request failed with status code: {response.status_code}")

    except requests.exceptions.RequestException as e:
        print(f"Error making API request: {e}")

    # Return default values in case of an exception
    return None

@app.route('/income', methods=['GET', 'POST'])
def income():
    result_df = get_data_from_api()

    if result_df is not None:
        predictions = []

        for turf__id in result_df['turf__id'].unique():
            turf_data = result_df[result_df['turf__id'] == turf__id]

            if not turf_data.empty:
                recent_row = turf_data.iloc[-1]

                try:
                    input_features = [
                        recent_row['income_LastWeek'],
                        recent_row['income_2Weeksback'],
                        recent_row['income_3Weeksback']
                    ]

                    predicted_income = linear_model.predict([input_features])[0]
                    rounded_predicted_income = round(predicted_income)

                except Exception as e:
                    print("There was an error during income prediction.")
                    # Set a default value for prediction
                    rounded_predicted_income = 2000  # Set your desired default value

                predictions.append({"turf__id": turf__id, "predicted_income": rounded_predicted_income})
            else:
                predictions.append({"turf__id": turf__id, "error": "No data found for the specified turf_id"})

        predictions_df = pd.DataFrame(predictions)

        return predictions_df.to_json(orient='records')
    else:
        print("Error fetching data from the API")
        return jsonify({"error": "Error fetching data from the API"})

if __name__ == '__main__':
    app.run(debug=True)
