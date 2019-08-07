#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 09:40:46 2019

@author: avneeshnolkha
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score,confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import pickle
import random
import pprint

r = random.randint
dataset = pd.read_csv('combined.csv')
le = LabelEncoder()

#Feature engineering for dashboard representation
def gender(temp):
    if temp == 0:
        x = 'Female'
    else:
        x = 'Male'
    return x

dataset['Customer_sex']=dataset['Customer_sex'].apply(gender)

def device_used(temp):
    if temp == 0:
        x = 'Mobile/Tablet'
    else:
        x = 'Laptop/PC'
    return x

dataset['Device_used']=dataset['Device_used'].apply(device_used)

def region(temp):
    list1 = 'Bangalore, Chennai, Delhi, Hyderabad, Kolkata, Mumbai,Visakhapatnam, Coimbatore, Ahmedabad, Pune, Surat'
    list1 = list1.replace(" ","")
    list1 = list1.split(',')
    
    if temp in list1:
        x = 'Metro'
    else:
        x = 'Non Metro'
    return x

dataset['Region']=dataset['Customer_city'].apply(region)

def states(temp):
    karnartaka=['Bangalore','Belgaum','Gulbarga','Hubli–Dharwad','Mangalore','Mysore','Bijapur']
    tamil=['Chennai','Erode','Madurai','Pondicherry','Vellore','Salem','Tiruchirappalli','Tirunelveli','Tiruppur',]
    delhi = ['Delhi']
    telangana=['Hyderabad','Warangal']
    bengal=['Kolkata','Asansol','Durgapur']
    maharashtra=['Mumbai','Pune','Amravati','Bhiwandi','Nagpur','Kolhapur','Nanded','Nashik','Sangli','Solapur','Vasai-VirarCity']
    andhra=['Visakhapatnam','Coimbatore','Guntur','Kakinada','Kurnool','Nellore','Rajahmundry','Tirupati','Tiruvannamalai','Vijayawada']
    gujrat=['Ahmedabad','Surat','Bhavnagar','Jamnagar','Rajkot','Vadodara']
    rajasthan=['Ajmer','Bikaner','Jaipur','Jodhpur','Kota']
    up=['Agra','Allahabad','Aligarh','Aurangabad','Bareilly','Ghaziabad','Jhansi','Kanpur','Lucknow','Mathura','Meerut','Moradabad','Noida','Allahbad','Varanasi','Firozabad']
    punjab=['Amritsar','Chandigarh','Jalandhar','Ludhiana','Ujjain']
    mp=['Bhopal','Gwalior','Indore','Jabalpur']
    orissa=['Bhubaneswar','Cuttack','Rourkela']
    jharkhand=['BokaroSteelCity','Dhanbad','Jamshedpur','Ranchi']
    chhattisgarh=['Bhilai','Raipur']
    uttrakhand=['Dehradun']
    haryana=['Faridabad','Gorakhpur','Gurgaon']
    assam=['Guwahati','Siliguri']
    kerela=['Kannur','Kochi','Kottayam','Kollam','Kozhikode','Malappuram','Palakkad','Thiruvananthapuram','Thrissur']
    jk=['Jammu','Srinagar']
    goa=['Goa']
    bihar=['Patna']
    
    if temp in karnartaka:
        x = 'Karnartaka'
    elif temp in tamil:
        x = 'Tamil Nadu'
    elif temp in delhi:
        x = 'Delhi'
    elif temp in telangana:
        x = 'Telangana'
    elif temp in bengal:
        x = 'West Bengal'
    elif temp in maharashtra:
        x = 'Maharashtra'
    elif temp in andhra:
        x = 'Andhra Pradesh'
    elif temp in gujrat:
        x = 'Gujrat'
    elif temp in rajasthan:
        x='Rajasthan'
    elif temp in up:
        x = 'Uttar Pradesh'
    elif temp in punjab:
        x = 'Punjab'
    elif temp in mp:
        x = 'Madhya Pradesh'
    elif temp in orissa:
        x = 'Orissa'
    elif temp in jharkhand:
        x = 'Jharkhand'
    elif temp in chhattisgarh:
        x = 'Chattisgarh'
    elif temp in uttrakhand:
        x = 'Uttrakhand'
    elif temp in haryana:
        x = 'Haryana'
    elif temp in assam:
        x = 'Assam'
    elif temp in kerela:
        x = 'Kerela'
    elif temp in jk:
        x = 'Jammu and Kashmir'
    elif temp in goa:
        x = 'Goa'
    elif temp in bihar:
        x = 'Bihar'
    else:
        x = 'Unknown'
    return x

dataset['State'] = dataset['Customer_city'].apply(states)

def zone(temp):  
    karnartaka=['Bangalore','Belgaum','Gulbarga','Hubli–Dharwad','Mangalore','Mysore','Bijapur']
    tamil=['Chennai','Erode','Madurai','Pondicherry','Vellore','Salem','Tiruchirappalli','Tirunelveli','Tiruppur',]
    delhi = ['Delhi']
    telangana=['Hyderabad','Warangal']
    bengal=['Kolkata','Asansol','Durgapur']
    maharashtra=['Mumbai','Pune','Amravati','Bhiwandi','Nagpur','Kolhapur','Nanded','Nashik','Sangli','Solapur','Vasai-VirarCity']
    andhra=['Visakhapatnam','Coimbatore','Guntur','Kakinada','Kurnool','Nellore','Rajahmundry','Tirupati','Tiruvannamalai','Vijayawada']
    gujrat=['Ahmedabad','Surat','Bhavnagar','Jamnagar','Rajkot','Vadodara']
    rajasthan=['Ajmer','Bikaner','Jaipur','Jodhpur','Kota']
    up=['Agra','Allahabad','Aligarh','Aurangabad','Bareilly','Ghaziabad','Jhansi','Kanpur','Lucknow','Mathura','Meerut','Moradabad','Noida','Allahbad','Varanasi','Firozabad']
    punjab=['Amritsar','Chandigarh','Jalandhar','Ludhiana','Ujjain']
    mp=['Bhopal','Gwalior','Indore','Jabalpur']
    orissa=['Bhubaneswar','Cuttack','Rourkela']
    jharkhand=['BokaroSteelCity','Dhanbad','Jamshedpur','Ranchi']
    chhattisgarh=['Bhilai','Raipur']
    uttrakhand=['Dehradun']
    haryana=['Faridabad','Gorakhpur','Gurgaon']
    assam=['Guwahati','Siliguri']
    kerela=['Kannur','Kochi','Kottayam','Kollam','Kozhikode','Malappuram','Palakkad','Thiruvananthapuram','Thrissur']
    jk=['Jammu','Srinagar']
    goa=['Goa']
    bihar=['Patna']
    
    if temp in karnartaka:
        x = 'South'
    elif temp in tamil:
        x = 'South'
    elif temp in delhi:
        x = 'North'
    elif temp in telangana:
        x = 'South'
    elif temp in bengal:
        x = 'East'
    elif temp in maharashtra:
        x = 'South'
    elif temp in andhra:
        x = 'South'
    elif temp in gujrat:
        x = 'West'
    elif temp in rajasthan:
        x='West'
    elif temp in up:
        x = 'North'
    elif temp in punjab:
        x = 'North'
    elif temp in mp:
        x = 'West'
    elif temp in orissa:
        x = 'East'
    elif temp in jharkhand:
        x = 'East'
    elif temp in chhattisgarh:
        x = 'East'
    elif temp in uttrakhand:
        x = 'North'
    elif temp in haryana:
        x = 'North'
    elif temp in assam:
        x = 'East'
    elif temp in kerela:
        x = 'South'
    elif temp in jk:
        x = 'North'
    elif temp in goa:
        x = 'South'
    elif temp in bihar:
        x = 'East'
    else:
        x = 'Unknown'
    return x

dataset['Zone']= dataset['Customer_city'].apply(zone)

data_features = dataset.loc[:,[2,3,6,24,25,8,9,10,11,18,19,]]

dataset['Price'].value_counts()
dataset['Product_comparison'] = 0

def compare(ctype):
    if ctype == 'Young':
        comparison = r(3,9)
    elif ctype == 'Middle':
        comparison = r(3,7)
    elif ctype == 'Old':
        comparison = r(1,4)
    return comparison

dataset['Product_comparison'] = dataset['Customer_type'].apply(compare)


def tech_score(row):
    score=0
    if row['Device_Platform'] == 'Linux':
        score = 5
    if row['Product_comparison'] < 4:
        score+= 2
    elif row['Product_comparison'] > 4 and row['Product_comparison'] < 7 :
        score+= 3 
    else:
        score+= 4
    return score
dataset['Tech_score'] = dataset.apply(tech_score, axis=1)
        
def purchase_power(row):
    power=0
    price = row['Price']
    platform = row['Device_Platform']
    if platform == 'iOS':
        power+= 4
    if price <45000:
        power+=3
    elif price >=45000 and price <70000:
        power+=5
    else:
        power+=6
    return power
dataset['Purchase_power'] = dataset.apply(purchase_power,axis=1)

def thinking_score(row):
    think = 0
    dinc = row['Days_in_cart']
    vinc = row['Viewd_in_cart']
    if dinc <11:
        think+= 1
    elif dinc >=11 and dinc<=30:
        think+=2
    elif dinc >30 and dinc <=60:
        think+=3
    else:
        think+=4
    if vinc <11:
        think+= 1
    elif vinc >=11 and vinc<=30:
        think+=2
    elif vinc >30 and dinc <=60:
        think+=3
    else:
        think+=4
    return think

dataset['Thinking_score'] = dataset.apply(thinking_score,axis=1)

dataset['Thinking_score'].value_counts()    
dataset['Purchase_power'].value_counts()   
dataset['Tech_score'].value_counts()

def package(row):
    think = row['Thinking_score']
    purchase = row['Purchase_power']
    tech=row['Tech_score']
    package='a'
    if purchase >=3 and purchase <=5:
        if tech <=4:
            package = 'Silver1'
        else:
            package = 'Silver2'
    elif purchase >=5 and purchase <=7:
        if tech <=4:
            package = 'Gold1'
        else:
            package = 'Gold2'
    elif purchase >7 and purchase <=10:
        if tech <=4:
            package = 'Platinum1'
        else:
            package = 'Platinum2'
    else :
        package = 'Undeterministic'
    return package

dataset['service_category'] = dataset.apply(package,axis=1)
   

#dataset.to_csv('final_wala.csv')      
  
columns = dataset.columns.tolist()  

features = dataset.iloc[:,[3,4,8,11,12,13,24,27]]
labels = dataset.iloc[:,-1]

#Now preparing Data For Machine Learning Modellinh
#Label Encoding

#gender
le1 = le
features['Customer_sex'] = le1.fit_transform(features['Customer_sex'])
#le2 -> customer_type
le2 = le
features['Customer_type'] = le2.fit_transform(features['Customer_type'])

#le3 -> Platform
le3 = le
features['Device_Platform']=le3.fit_transform(features['Device_Platform'])

#le4 -> Labels(service category)
le4 = le
labels=le4.fit_transform(labels)

#le5 -> Region
le5 = le
features['Region'] = le5.fit_transform(features['Region'])


scaler = StandardScaler()
features = scaler.fit_transform(features)

x_train , x_test , y_train , y_test = train_test_split(features, labels , test_size = 0.2 , random_state=42)

models = []

models.append(('Logistic', LogisticRegression()))
models.append(('Linear Discriminant Analysis', LinearDiscriminantAnalysis()))
models.append(('KNN Classifier', KNeighborsClassifier()))
models.append(('DecisionTree', DecisionTreeClassifier()))
models.append(('Support Vector Classifier', SVC()))
models.append(('XG Boost Classifier', XGBClassifier()))
models.append(('Random Forest', RandomForestClassifier(n_estimators=50)))


test_score = pd.DataFrame([])
for name, model in models:
    model.fit(x_train,y_train)
    labels_pred=model.predict(x_test)
    score = model.score(x_test,y_test)
    test_score=test_score.append(pd.DataFrame({'Model':name,'Score':score},index=[0]), ignore_index=True)
    
test_score.to_csv('test_score.csv')

"""As random forest is giving us the highest amount of accuracy, we will go for random forest and improve accuracy by using hyperparameter tuning"""
rf = RandomForestClassifier(random_state=0,n_estimators=50)
print('Parameters currently in use:\n')
print(rf.get_params())


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)# Fit the random search model
rf_random.fit(x_train, y_train)

rf_random.best_params_
#Predicting using the best parameters given
rf = RandomForestClassifier(n_estimators= 1600,min_samples_split= 5,min_samples_leaf = 1,max_features ='auto',max_depth =10,bootstrap= True)

rf.fit(x_train,y_train)
y_pred=rf.predict(x_test)
score = rf.score(x_test,y_test)

filename = 'clf.pkl'
pickle.dump(rf, open(filename, 'wb'))






















