from flask import Flask, jsonify
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import requests


app = Flask(__name__)




@app.route('/sentiment', methods=['GET','POST'])
def sentiment():
    api_url = "http://192.168.1.21:9000/user/turf_rating/"
    response = requests.get(api_url)
    if response.status_code == 200:
        data = response.json()
    
    df = pd.DataFrame(data, columns=['turfid', 'rating', 'review'])
    
    nltk.download('vader_lexicon')
    
    # Create a SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()

    # Add a new column 'sentiment' to the DataFrame
    df['sentiment'] = df['review'].apply(lambda x: 'positive' if sid.polarity_scores(x)['compound'] >= 0 else 'negative')

    # df['weighted_rating'] = df.apply(lambda row: row['rating'] * (1.2 if row['sentiment'] == 'positive' else 0.8), axis=1)
    df['weighted_rating'] = df.apply(lambda row: min(row['rating'] * (1.2 if row['sentiment'] == 'positive' else 0.8), 5), axis=1)
    # df['weighted_rating'] = df.apply(lambda row: round(min(row['rating'] * (1.2 if row['sentiment'] == 'positive' else 0.8), 5), 1), axis=1)
    # df['weighted_rating']=round(df['weighted_rating'],1)
 
 
    # Calculate the overall rating for each turf
    overall_rating = round(df.groupby('turfid')['weighted_rating'].mean().reset_index(),1)

    # Convert the result to JSON
    result = overall_rating.to_dict(orient='records')

    return result

if __name__ == '__main__':
    app.run(debug=True)
