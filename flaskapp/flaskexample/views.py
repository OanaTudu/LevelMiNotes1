from flask import render_template
from flaskexample import app
from flaskexample.a_Model import ModelIt
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import pandas as pd
import psycopg2
from flask import Flask, request, jsonify, flash, url_for, redirect
import numpy as np
import re
import pickle
import music21
from music21 import *
from werkzeug import secure_filename
import os
from flask import send_from_directory
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix




# Python code to connect to Postgres
# You may need to modify this based on your OS,
# as detailed in the postgres dev setup materials.
user = 'oana' #add your Postgres username here
host = 'localhost'
dbname = 'birth_db'
db = create_engine('postgres://%s%s/%s'%(user,host,dbname))
con = None
con = psycopg2.connect(database = dbname, user = user)
ALLOWED_EXTENSIONS = set(['xml', 'mxl', 'capx', 'abc', 'krn'])


@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html",
       title = 'Home', user = { 'nickname': 'Oana' },
       )

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory( '/Users/oana/work/insight/flaskapp/uploaded',
                               filename)

@app.route('/api',methods=['POST'])
def predict():
    data = request.get_json(force=True)
    model = ModelIt()
    prediction = model.predict([[np.array(data['exp'])]])
    output = prediction[0]
    return jsonify(output)

@app.route('/output')
def model_output(filename):
    #load the model
    model = ModelIt()
    model = pickle.load(open('model.pkl','rb'))
    #model = ModelIt()
    #read the file and extract the features:
    def featureExtractor(filename):

        #parse file
        p = converter.parse(filename)
        key = p.analyze('key') #returns the key
        chord = p.chordify() #returns the instrument and a stream object to extract other features
        d = chord.duration
        #duration of piece
        dur = str(d)
        digits = r"\d+"
        d_dig = re.findall(digits,dur)
        duration = float(d_dig[1])+float(d_dig[2])/(10*len(d_dig[2]))
        init_ts = p.recurse().getElementsByClass(meter.TimeSignature)[0]  #initial time signature
        it = str(init_ts)
        it_dig = re.findall(digits,it)
        initialTimeSig = it_dig[1] + '/' + it_dig[2]
        if len(set(p.recurse().getElementsByClass(meter.TimeSignature))):
            changeTimeSig = 0
        else:
            changeTimeSig = 1
        highestOffset = chord.highestOffset #highest offset
        #highest of all element offsets plus duration
        highestTimeD=chord.highestTime
        #melInt = p.melodicIntervals()
        beatDuration = p.beatDuration
        instrument= str(chord.getInstruments())[21:-1]
        feats = [duration, highestOffset, highestTimeD]

        return feats
    #get features from uploaded file
    fName = os.path.join('/Users/oana/work/insight/flaskapp/uploaded', filename)
    features = featureExtractor(fName)
    #there is only one sample, so reshape:
    features = np.array(features).reshape(1, -1)
    y_pred_file = model.predict(features)
    print('Test set predictions: \n{}'.format(y_pred_file))
    y_prob = model.predict_proba(features)[:, 1]
    print('Probability for predictions: \n{}'.format(y_prob))

    #return render_template("output.html", predictedDifficulty = y_pred_file, predictionProba = y_prob)
    return  y_pred_file, y_prob

#app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload_file', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join('/Users/oana/work/insight/flaskapp/uploaded', filename))
            y_pred_file, y_prob = model_output(filename)
            return render_template("output.html", predictedDifficulty = y_pred_file, predictionProba = y_prob)


    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''
