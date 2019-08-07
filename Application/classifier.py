#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 14:23:41 2019

@author: avneeshnolkha
"""

import pandas as pd
import numpy as np
import seaborn as sns
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score,confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

"""
    
"""

le = LabelEncoder()


data = pd.read_csv('new_data.csv')
columns = data.columns.tolist()
#Data Preprocessingzz
#le1 -> Gender (0 -> Female , 1-> male)
le1 = le 
data['Gender']=le1.fit_transform(data['Gender'])

#le2 -> Device used(0-> Laptop 1-> Mobile)
le2 = le
data['Device_used']=le2.fit_transform(data['Device_used'])

#le3 -> Device_platform
le3 = le
data['Device_Platform']=le3.fit_transform(data['Device_Platform'])

#le4 -> City
le4 = le
data['City']=le4.fit_transform(data['City'])

#le5 -> Warranty
le5 = le
data['Warranty']=le5.fit_transform(data['Warranty'])

#le6 -> Support Given
le6 = le
l1 = ['Yes','No']
data['Support_given'] = le6.fit(l1).transform(data['Support_given'])

#le7 -> Support_Type
le7 = le
data['Support_type'] = le7.fit_transform(data['Support_type'])

#le8 -> Support_mode
le8 = le
data['Support_mode'] = le8.fit_transform(data['Support_mode'])

#le9 -> Region
le9 = le
data['Region'] = le9.fit_transform(data['Region'])

#le10 -> state
le10 = le
data['State'] = le10.fit_transform(data['State'])

#le11  -> Zone
le11 = le
data['Zone'] = le11.fit_transform(data['Zone'])

data = data.iloc[:,3:]
columns = data.columns.to_list()

#Creating Labels
label1 = data['Support_type']
label2 = data['Support_mode']
#Creating Features
features = data
features = features.drop('Support_type',axis=1)
features = features.drop('Support_mode',axis=1)
dt_features_name = features.columns.to_list()
#standard Scaling
scale = StandardScaler()
features = scale.fit_transform(features.values)

#Classifier 1 -> support type
x_train1 , x_test1 , y_train1,y_test1 = train_test_split(features,label1,test_size=0.15,random_state=0)

classifier1 = DecisionTreeClassifier()
classifier1.fit(x_train,y_train)
y_pred = classifier1.predict(x_test)

cm = confusion_matrix(y_test,y_pred)

"""dt_target_names = [str(s) for s in data['Support_type']]
export_graphviz(classifier1, out_file='tree.dot', 
    feature_names=dt_features_name, class_names=dt_target_names,
    filled=True)  
graph = pydotplus.graph_from_dot_file('tree.dot')
Image(graph.create_png())"""

# save the model to disk
filename1 = 'clf1_type.sav'
pickle.dump(classifier1, open(filename1, 'wb'))

#Classifier 2 -. support_mode
x_train2 , x_test2 , y_train1, y_test2 = train_test_split(features,label2,test_size=0.15,random_state=0)

classifier2 = DecisionTreeClassifier()
classifier2.fit(x_train,y_train)
y_pred = classifier2.predict(x_test)

cm2 = confusion_matrix(y_test,y_pred)

#Saving the model to disk
filename2 = 'clf2_mode.sav'
pickle.dump(classifier2, open(filename2, 'wb'))

#Loading the model

def clf1(x):
    model1 = pickle.load(open(filename1, 'rb'))
    result1 = model1.score(x_test, y_test)
    prediction1 = model1.predict(x)
    print (result1)
    return prediction1
    
def clf2():
    model2 = pickle.load(open(filename2, 'rb'))
    result2 = model2.score(x_test, y_test)
    prediction2 = model2.predict(x)
    print (result2)
    return prediction2
    

































































































