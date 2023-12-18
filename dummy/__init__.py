from flask import Flask, jsonify
import joblib
import pandas as pd
import numpy as np
import requests
import json

app = Flask(__name__)



@app.route('/sentiment', methods=['GET', 'POST'])
def income():

        return jsonify({"message":"nidha_code"})

if __name__ == '__main__':
    app.run(debug=True)
