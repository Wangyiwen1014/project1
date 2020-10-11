
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score  #For accuracy calculation
from sklearn.tree import DecisionTreeClassifier #Import Decision Tree Classifier
from sklearn.model_selection import train_test_split #Import train_test_split function

col_names = ['winner','gameId','creationTime', 'gameDuration', 'seasonId', 
             'firstBlood', 'firstTower', 'firstInhibitor', 'firstBaron',
             'firstDragon','firstRiftHerald','t1_towerKills','t1_inhibitorKills',
             't1_baronKills','t1_dragonKills','t1_riftHeraldKills',
             't2_towerKills','t2_inhibitorKills','t2_baronKills',
             't2_dragonKills','t2_dragonKills']

# load dataset
nd = pd.read_csv('new_data.csv')#training set
nd = nd.iloc[1:] # delete the first row of the dataframe

ts = pd.read_csv('test_set.csv')#testing set
ts = ts.iloc[1:] # delete the first row of the dataframe


#split dataset in features and target variable
feature_cols = ['firstBlood', 'firstTower', 'firstInhibitor', 'firstBaron',
                'firstDragon','firstRiftHerald','t1_towerKills',
                't1_inhibitorKills','t1_baronKills','t1_dragonKills',
                't1_riftHeraldKills','t2_towerKills','t2_inhibitorKills',
                't2_baronKills','t2_dragonKills','t2_dragonKills']

X_train = nd[feature_cols] # Features
y_train = nd.winner # Target variable

X_test = ts[feature_cols] # Features
y_test = ts.winner # Target variable


# Create Decision Tree classifer object 
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer 
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset 
y_pred = clf.predict(X_test)

# Model Accuracy 
print("Accuracy:",accuracy_score(y_test, y_pred)) 

