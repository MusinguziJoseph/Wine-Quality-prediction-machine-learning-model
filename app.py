from flask import Flask,render_template ,jsonify,request
import joblib
import numpy as np


#load the machine learning model
clf = joblib.load(open('joblib_model.sav',"rb"))

#create a flask app
app = Flask(__name__)

#Define the route to render the html GUI
@app.route('/')
def home():
    return render_template('home.html')

#Define a route to predict the wine_quality
@app.route('/predict',methods=['POST'])
def predict():
    fixed_acidity=float(request.form['fixed_acidity'])
    volatile_acidity=float(request.form['volatile_acidity'])
    citric_acid=float(request.form['citric_acid'])
    residual_sugar=float(request.form['residual_sugar'])
    chlorides=float(request.form['chlorides'])
    free_sulfur_dioxide=float(request.form['free_sulfur_dioxide'])
    total_sulfur_dioxide=float(request.form['total_sulfur_dioxide'])
    ph=float(request.form['ph'])
    sulphates=float(request.form('sulphates'))
    alcohol=float(request.form['alcohol'])

    #predict the wine quality
    prediction=clf.predict([
        fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,
        total_sulfur_dioxide,sulphates,alcohol])
    
    #return the prediction
    return render_template('result.html',prediction=prediction)

#run the flask app
if __name__ == '_main_':


      app.run(debug=True)
