from flask import Flask, render_template, session, redirect, url_for
from flask_wtf import FlaskForm
from wtforms import TextField,SubmitField,SelectField
from wtforms.validators import NumberRange
from nbm import *
import numpy as np
import pandas as pd
import json

ATTR_NEEDED = ['Location','MinTemp','MaxTemp','Rainfall','Evaporation','Sunshine','WindGustDir','WindGustSpeed','WindDir9am','WindDir3pm','WindSpeed9am','WindSpeed3pm','Humidity9am','Humidity3pm','Pressure9am','Pressure3pm','Cloud9am','Cloud3pm','Temp9am','Temp3pm','RainToday']
LOCATION_LST = ['Albury','BadgerysCreek','Cobar','CoffsHarbour','Moree','Newcastle','NorahHead','NorfolkIsland','Penrith','Richmond','Sydney','SydneyAirport','WaggaWagga','Williamtown','Wollongong','Canberra','Tuggeranong','MountGinini','Ballarat','Bendigo','Sale','MelbourneAirport','Melbourne','Mildura','Nhil','Portland','Watsonia','Dartmoor','Brisbane','Cairns','GoldCoast','Townsville','Adelaide','MountGambier','Nuriootpa','Woomera','Albany','Witchcliffe','PearceRAAF','PerthAirport','Perth','SalmonGums','Walpole','Hobart','Launceston','AliceSprings','Darwin','Katherine','Uluru']
WIND_DIR_LST = ['SW','NW','S','NNE','N','ESE','SE','ENE','E','NE','NNW','SSE','W','WNW','SSW','WSW']

MODEL_NAME = 'Aus_weather_mod'

def pred(model, input_json):
    
    # Create X_test from input data
    input_lst = []
    for i in ATTR_NEEDED:
        input_lst.append(input_json[i])
    X_test = pd.DataFrame(np.array(input_lst).reshape(1,-1))
    y_pred = []

    probs = {k: np.log(v) for k, v in model.prior.items()}
    # Loop through each attribute column
    for col in range(len(model.attr_types)):
        value = X_test.iloc[0, col]
            
        if not (isinstance(value, float)
                and np.isnan(value)): # Ignoring missing values
            # Update the priors with the log-transformed of likelihoods
            
            # Nominal & Ordinal Case
            if (model.attr_types[col] == 0 or model.attr_types[col] == 1): 
                for label in model.labels:
                    if value in model.params[col][label].keys(): # Ignore unseen attributes
                        probs[label] += np.log(model.params[col][label][value])
            # Numeric Case
            elif (model.attr_types[col] == 2):
                value = float(value)
                for label in model.labels:
                    gaussian_prob = gaussian_pdf(value,
                                                 model.params[col][label]["mu"],
                                                 model.params[col][label]["sigma"])
                    if gaussian_prob != 0:
                        probs[label] += np.log(gaussian_prob)
    # ArgMax
    y_pred.append(max(probs, key=probs.get))
    
    return y_pred[0]

app = Flask(__name__)
# Configure a secret SECRET_KEY
app.config['SECRET_KEY'] = 'someRandomKey'

# Load model
model = NaiveBayesModel
model.load_model(model, MODEL_NAME)

# Now create a WTForm Class
class WeatherForm(FlaskForm):
    Location = SelectField(label='Location', choices=[(i,i) for i in LOCATION_LST])
    MinTemp = TextField('MinTemp (-8.5, 33.9)')
    MaxTemp = TextField('MaxTemp (-4.8, 48.1)')
    Rainfall = TextField('Rainfall (0.0, 371.0)')
    Evaporation = TextField('Evaporation (0.0, 145.0)')
    Sunshine = TextField('Sunshine (0.0, 14.5)')
    WindGustDir = SelectField(u'WindGustDir', choices=[(i,i) for i in WIND_DIR_LST])
    WindGustSpeed = TextField('WindGustSpeed (6.0, 135.0)')
    WindDir9am = SelectField(u'WindDir9am', choices=[(i,i) for i in WIND_DIR_LST])
    WindDir3pm = SelectField(u'WindDir3pm', choices=[(i,i) for i in WIND_DIR_LST])
    WindSpeed9am = TextField('WindSpeed9am (0.0, 130.0)')
    WindSpeed3pm = TextField('WindSpeed3pm (0.0, 87.0)')
    Humidity9am = TextField('Humidity9am (0.0, 100.0)')
    Humidity3pm = TextField('Humidity3pm (0.0, 100.0)')
    Pressure9am = TextField('Pressure9am (980.5, 1041.0)')
    Pressure3pm = TextField('Pressure3pm (977.1, 1039.6)')
    Cloud9am = TextField('Cloud9am (0.0, 9.0)')
    Cloud3pm = TextField('Cloud3pm (0.0, 9.0)')
    Temp9am = TextField('Temp9am (-7.2, 40.2)')
    Temp3pm = TextField('Temp3pm (-5.4, 46.7)')
    RainToday = SelectField(u'RainToday', choices=[('Yes', 'Yes'),('No','No')])
    submit = SubmitField('Forecast Weather')

@app.route('/', methods=['GET', 'POST'])
def index():
    # Create instance of the form.
    form = WeatherForm()
    # If the form is valid on submission
    if form.validate_on_submit():
        # Grab the data from the input on the form.
        session['Location'] = form.Location.data
        session['MinTemp'] = form.MinTemp.data
        session['MaxTemp'] = form.MaxTemp.data
        session['Rainfall'] = form.Rainfall.data
        session['Evaporation'] = form.Evaporation.data
        session['Sunshine'] = form.Sunshine.data
        session['WindGustDir'] = form.WindGustDir.data
        session['WindGustSpeed'] = form.WindGustSpeed.data       
        session['WindSpeed3pm'] = form.WindSpeed3pm.data
        session['WindSpeed9am'] = form.WindSpeed9am.data
        session['WindDir3pm'] = form.WindDir3pm.data
        session['WindDir9am'] = form.WindDir9am.data       
        session['Humidity9am'] = form.Humidity9am.data
        session['Humidity3pm'] = form.Humidity3pm.data
        session['Pressure9am'] = form.Pressure9am.data
        session['Pressure3pm'] = form.Pressure3pm.data        
        session['Cloud9am'] = form.Cloud9am.data
        session['Cloud3pm'] = form.Cloud3pm.data
        session['Temp9am'] = form.Temp9am.data
        session['Temp3pm'] = form.Temp3pm.data
        session['RainToday'] = form.RainToday.data
        return redirect(url_for('prediction'))
    return render_template('home.html', form=form)

@app.route('/prediction')
def prediction():
    #Defining content dictionary
    content = {}
    content['Location'] = str(session['Location'])
    content['MinTemp'] = float(session['MinTemp'])
    content['MaxTemp'] = float(session['MaxTemp'])
    content['Rainfall'] = float(session['Rainfall'])
    content['Evaporation'] = float(session['Evaporation'])
    content['WindGustDir'] = str(session['WindGustDir'])
    content['WindGustSpeed'] = float(session['WindGustSpeed'])       
    content['WindSpeed3pm'] = float(session['WindSpeed3pm'])
    content['WindSpeed9am'] = float(session['WindSpeed9am'])
    content['WindDir3pm'] = str(session['WindDir3pm'])
    content['WindDir9am'] = str(session['WindDir9am'])      
    content['Humidity9am'] = float(session['Humidity9am'])
    content['Humidity3pm'] = float(session['Humidity3pm'])
    content['Pressure9am'] = float(session['Pressure9am'])
    content['Pressure3pm'] = float(session['Pressure3pm'])        
    content['Cloud9am'] = float(session['Cloud9am'])
    content['Cloud3pm'] = float(session['Cloud3pm'])
    content['Temp9am'] = float(session['Temp9am'])
    content['Temp3pm'] = float(session['Temp3pm'])
    content['RainToday'] = str(session['RainToday'])
    content['Sunshine'] = float(session['Sunshine'])
    results = pred(model, input_json=content)
    return render_template('prediction.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)