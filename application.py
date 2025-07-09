from flask import Flask, request, jsonify, render_template

import pickle 
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler 

application = Flask(__name__)
app=application


# import ridge regressor model and standard scaler pickle file
ridge_model = pickle.load(open("models/ridge.pkl", "rb"))
standard_scaler = pickle.load(open("models/scaler.pkl", "rb"))


@app.route("/")
def index(): 
    return render_template('index.html')


@app.route("/predict", methods=['GET', 'POST'])
def predict_datapoints(): 
    if request.method == 'POST': 
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        new_scaled_features = standard_scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
        prediction = ridge_model.predict(new_scaled_features)

        return render_template('home.html', prediction=prediction[0])
    else: 
        return render_template('home.html')
    

if __name__ == "__main__": 
    app.run(host="0.0.0.0")