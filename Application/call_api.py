#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 14:58:05 2019

@author: avneeshnolkha
"""

import requests
import json
import pandas as pd

def gen_url():
    
    df = pd.read_excel('testdata.xlsx')
    l=[]
    
    for i, r in df.iterrows():
        l1=[]
        v = r['Customer_sex']
        if v == "Female":
            l1.append(0)
        else:
            l1.append(1)
        l1.append(r['Customer_type'])
        l1.append(r['Device_Platform'])
        l1.append(r['product_visit'])
        l1.append(r['Days_in_cart'])
        l1.append(r['Viewd_in_cart'])
        l1.append(r['Region'])
        l1.append(r['Product_comparison'])
        l.append(l1)
        
    print("Here..")
    url = 'http://127.0.0.1:5000/api/'
    
    data = l
    j_data = json.dumps(data)
    headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
    print("Here too!")
    print("URL = ", url)
    r = requests.post(url, data=j_data, headers=headers)
    print(r, r.text)
    return r