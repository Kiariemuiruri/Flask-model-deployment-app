import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
import pickle
import sklearn
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score
import pandas as pd

# create a flask app
app = Flask(__name__, template_folder='template')

# load the pickle model
model = pickle.load(open("Scripts/model.pkl", "rb"))
# input preprocesing
scaler = StandardScaler()
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)

@app.route("/")
def Home():
    return render_template('base.html')
@app.route("/predict", methods=['POST'])
def predict():
    # extract form data
    amount = float(request.form['Amount'])     
    app_income = float(request.form['AppIncome'])
    coapp_income = float(request.form['CoappIncome'])
    property_area = request.form['PropertyArea']
    credict_history = float(request.form['CreditHistory'])
    self_employed = request.form['SelfEmployed']

    # create a dictionary to fold the data
    data = {
        'Amount':[amount],
        'AppIncome':[app_income],
        'CoappIncome':[coapp_income],
        'PropertyArea':[property_area],
        'CreditHistory':[credict_history],
        'SelfEmployed':[self_employed]
    }
    df = pd.DataFrame(data)
    df['PropertyArea'] = df['PropertyArea'].map({'Urban': 0, 'Rural':1, 'Semiurban':2})
    df['SelfEmployed'] = df['SelfEmployed'].map({'Yes':1, 'No':0})

    #float_features = [float(x) for x in request.form.values()]
    #features = [np.array(float_features)]
    features_scaled = scaler.fit_transform(df)
    features_scaled_poly = poly.fit_transform(features_scaled)
    prediction = model.predict(features_scaled_poly)
    
    for pred in prediction:
        if pred == 1:
            pred = 'Approved'
        elif pred == 0:
            pred = 'Declined'
    return render_template('base.html', prediction_text="Your Loan has been {}!".format(pred))
    
# def options():
# create dummies for string inputs

        
if __name__=='__main__':
    app.run(debug=True)