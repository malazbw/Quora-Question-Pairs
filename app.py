"""
@author: malazbw
"""
import os
import sys
import logging
from flask import Flask, request, jsonify, send_file,render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
from quora_model import QuoraModel
import tensorflow as tf

# define the app
app = Flask(__name__)

# load the model
model = QuoraModel()
    
@app.route('/home_page', methods=['POST','GET'])
def login():

    if request.method == "POST":
        q1 = request.form["q1"]
        q2 = request.form["q2"]
        output_data = model.predict(q1, q2)
        return f"<h1>{output_data}</h1>"
    else:
        return render_template("home_page.html")


# HTTP Errors handlers
@app.errorhandler(404)
def url_error(e):
    return """
    Wrong URL!
    <pre>{}</pre>""".format(e), 404


@app.errorhandler(500)
def server_error(e):
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500

def predict(input1,input2):
    return QuoraModel.pred(input1,input2)


if __name__ == '__main__':
    # This is used when running locally.
    app.debug = True

    app.run(host='0.0.0.0')
