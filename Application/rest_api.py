#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 14:45:28 2019

@author: avneeshnolkha
"""

from flask import Flask, request, redirect, url_for, flash, jsonify
import numpy as np
import pickle as p
import json
from pip._internal import main
from call_api import gen_url



app = Flask(__name__)


@app.route('/api/', methods=['POST'])
def makecalc1():
    data = request.get_json()
    prediction1 = np.array2string(model1.predict(data))
    print("prediction => ",prediction1)
    return jsonify(prediction1)

def callRestApi():
    print("Calling...")
    modelfile1 = 'clf.pkl'
    model1 = p.load(open(modelfile1, 'rb'))
    return gen_url()