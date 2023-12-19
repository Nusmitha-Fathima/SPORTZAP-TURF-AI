from flask import Flask, jsonify
import pandas as pd
from sklearn.ensemble import IsolationForest
import requests

app = Flask(__name__)
@app.route('/send_notifications', methods=['GET'])
def send_notifications():
    api_url = "https://8882-116-68-110-250.ngrok-free.app/admin_app/monthly_turf_income/"
    response = requests.get(api_url)
    df = pd.DataFrame()
    if response.status_code == 200:
        data = response.json()
        message = data['message']
        df = pd.DataFrame(message)
    selected_columns = ['turf__id','turf__price' ,'turf__owner','total_income']
    df_selected = df[selected_columns]
    df = df_selected[['turf__price']]
    model = IsolationForest(random_state=42) 
    model.fit(df)
    df_selected['Outlier'] = model.predict(df)
    outliers_df = df_selected[df_selected['Outlier'] == -1]
    print(outliers_df)
    inliers_df = df_selected[df_selected['Outlier'] == 1]
    min_value = inliers_df['total_income'].min()
    owner_notification = outliers_df[outliers_df["total_income"] < min_value]
    notification_message = "Notification: Your turf's monthly earnings are relatively lower compared to other turfs. Consider implementing specific measures such as promoting events, adjusting pricing, or enhancing the turf facilities to attract more customers and increase earnings. If needed, feel free to discuss strategies with the management team. Thank you for your attention and efforts to improve your turf's performance."
    owner_id = "none"
    for index, row in owner_notification.iterrows():
        owner_id = row['turf__owner']
    return jsonify({'message': notification_message, 'id': owner_id})


if __name__ == '__main__':
    app.run(debug=True)
