from flask import Flask, jsonify
import pandas as pd
import requests

app = Flask(__name__)
@app.route('/', methods=['GET'])

def welcome():

    return jsonify({'message': "welcome to ai applications"})


if __name__ == '_main_':
    app.run(debug=True)