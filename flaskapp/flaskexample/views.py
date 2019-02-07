from flask import render_template
from flaskexample import app
from flaskexample.a_Model import ModelIt
import pandas as pd
import seaborn as sns
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
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt




# Python code to connect to Postgres
# You may need to modify this based on your OS,
# as detailed in the postgres dev setup materials.

ALLOWED_EXTENSIONS = set(['xml', 'mxl', 'abc', 'krn'])





@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory( '/home/ubuntu/flaskapp/uploaded',
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
    #Extract features from file:

    def featureExtractor(composer, filename):
        ti=filename#os.path.basename()
        #parse file
        p = converter.parse(filename)
        quarterLength = p.metadata.quarterLength
        #print(quarterLength)
        key = str(p.analyze('key')) #returns the key
        acc = ['C major a minor','G major e minor F major d minor','D major b minor B- major g minor',
       'A major f# minor E- major c minor','E major c# minor A- major f minor',
       'D- major b- minor B major g# minor','F# major d# minor G- major e- minor',
       'C# major a# minor C- major a- minor']

        keySig = [2**x for x in np.arange(len(acc)) if re.findall(key,acc[x])]
        keySig = keySig[0]
        chord = p.chordify() #returns the instrument and a stream object to extract other features
        d = chord.duration

        #duration of piece
        dur = str(d)
        digits = r"\d+"
        d_dig = re.findall(digits,dur)
        duration = float(d_dig[1])+float(d_dig[2])/(10*len(d_dig[2]))
        init_ts = p.recurse().getElementsByClass(meter.TimeSignature)[0]  #initial time signature
        changeKeySig = 1
        #print(init_ts)
        it = str(init_ts)
        it_dig = re.findall(digits,it)
        initialTimeSig = it_dig[1] + '/' + it_dig[2]
        initTimeSig = float(it_dig[1])/float(it_dig[2])
        changeTimeSig = len(p.recurse().getElementsByClass(meter.TimeSignature))
        highestOffset = chord.highestOffset #highest offset
        lowestOffset = chord.lowestOffset   #lowestOffset
        #highest of all element offsets plus duration
        highestTimeD=chord.highestTime
        #melInt = p.melodicIntervals()
        beatDuration = p.beatDuration
        instrument= str(chord.getInstruments())[21:-1]
        #if not noteCount:
        noteCount = duration/initTimeSig
        #print(f'Composer: {composer}, title: {ti}')
        feats = [composer, ti, keySig, duration,  highestOffset, highestTimeD, quarterLength, initTimeSig, changeTimeSig, noteCount, changeKeySig]
        #print(feats)
        return feats
    #get features from uploaded file
    fName = os.path.join('/home/ubuntu/flaskapp/uploaded', filename)
    toks = re.split('_', os.path.basename(fName))
    if len(toks) > 1:
        cmp = toks[0]
    else:
        cmp = 'bach'
    features = featureExtractor(cmp, fName)
    #eliminate composer and title fields
    features = features[2:]
    #there is only one sample, so reshape:
    features = np.array(features).reshape(1, -1)
    y_pred_file = model.predict(features)
    y_prob = model.predict_proba(features)
    #return render_template("output.html", predictedDifficulty = y_pred_file, predictionProba = y_prob)
    return  y_pred_file, y_prob

@app.route('/test', methods=['POST'])
def chartTest(prob_array, fName):
    pred_prec = np.insert(prob_array[0], 0, 0)
    df_Pred = pd.DataFrame({'Accuracy': pred_prec}, index=np.arange(1,9))
    #picName = fName[:-4] + 'plot.png'

    picName = 'new_' + 'plot.png'
    myfile = os.path.join('/home/ubuntu/flaskapp/flaskexample/static/images', picName)
    plt.cla()
    if os.path.isfile(myfile):
        os.unlink(myfile)
    df_wide=df_Pred.transpose()
    sns.set(font_scale=1.4)
    ax = sns.heatmap(df_wide,  annot=True, annot_kws={"size":14}, fmt = '.1%', linewidth=2., cmap = 'Blues', square = True, cbar = False)
    plt.savefig(myfile,  bbox_inches='tight')
    plt.cla()
    return picName


#app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/' ,  methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
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
            file.save(os.path.join('/home/ubuntu/flaskapp/uploaded', filename))
            y_pred_file, y_prob = model_output(filename)
            pic = chartTest(y_prob, filename)
            pic = '/images/'+pic
            #fig = create_figure(pred_prec)
            return render_template("output.html", predictedDifficulty = y_pred_file[0], predictionProba = y_prob, fName = filename, picFile = pic)


    return render_template("index.html")
