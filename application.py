import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))  # this is the trained scaler

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST', 'GET'])
def predictfn():
    result = None
    if request.method == 'POST':
        Temperature = float(request.form['Temperature'])
        RH = float(request.form['RH'])
        Ws = float(request.form['Ws'])    
        Rain = float(request.form['Rain'])
        FFMC = float(request.form['FFMC'])        
        DMC = float(request.form['DMC'])    
        ISI = float(request.form['ISI'])        
        Classes = float(request.form['Classes'])
        Region = float(request.form['Region'])

        input_data = [[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]]
        scaled_input = scaler.transform(input_data)
        result = ridge_model.predict(scaled_input)

    return render_template('index.html', results=result[0] if result is not None else None)

if __name__ == "__main__":
    app.run(debug=True)
