from flask import Flask, render_template, request, redirect, url_for, jsonify
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
application = Flask(__name__)
app = application

scaler = pickle.load(open('models/scaler.pkl', 'rb'))
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/preditdata", methods =['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        # Get form values
        Temperature = float(request.form["Temperature"])
        RH = float(request.form["RH"])
        Ws = float(request.form["Ws"])
        Rain = float(request.form["Rain"])
        FFMC = float(request.form["FFMC"])
        DMC = float(request.form["DMC"])
        ISI = float(request.form["ISI"])
        Classes = request.form["Classes"]
        Region = request.form["Region"]


        new_data_scaled = scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
        return render_template('home.html', result = ridge_model.predict(new_data_scaled)[0])
    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
