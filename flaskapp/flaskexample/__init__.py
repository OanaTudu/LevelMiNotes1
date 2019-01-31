from flask import Flask
import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
app = Flask(__name__)
from flaskexample import views


UPLOAD_FOLDER = '/work/insight/flaskapp/uploaded'
ALLOWED_EXTENSIONS = set(['xml', 'mxl', 'capx', 'abc', 'krn'])
