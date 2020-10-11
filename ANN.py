import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score #for accuracy calculation


col_names = ['winner','firstBlood', 'firstTower', 'firstInhibitor', 'firstBaron',
             'firstDragon','firstRiftHerald','t1_towerKills','t1_inhibitorKills',
             't1_baronKills','t1_dragonKills','t1_riftHeraldKills',
             't2_towerKills','t2_inhibitorKills','t2_baronKills',
             't2_dragonKills','t2_dragonKills']

#load data
nd = pd.read_csv('new_data.csv')#training set
ts = pd.read_csv('test_set.csv')#testing set

#Remaping
mappings = {
    1:0,
    2:1
}
nd['winner'] = nd['winner'].apply(lambda x: mappings[x])
ts['winner'] = ts['winner'].apply(lambda x: mappings[x])


#Split dataset into features and target
feature_cols = ['firstBlood', 'firstTower', 'firstInhibitor', 'firstBaron',
                'firstDragon','firstRiftHerald','t1_towerKills',
                't1_inhibitorKills','t1_baronKills','t1_dragonKills',
                't1_riftHeraldKills','t2_towerKills','t2_inhibitorKills',
                't2_baronKills','t2_dragonKills','t2_dragonKills'] 

X_train = nd[feature_cols].values # Features
y_train = nd.winner.values # Target variable

X_test = ts[feature_cols].values # Features
y_test = ts.winner.values # Target variable


#Convert split data from Numpy arrays to PyTorch tensors
X_train = torch.FloatTensor(X_train) 
X_test = torch.FloatTensor(X_test) 
y_train = torch.LongTensor(y_train) 
y_test = torch.LongTensor(y_test)

#ANN model declaration
class ANN(nn.Module): 
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=16, out_features=100) 
        self.output = nn.Linear(in_features=100, out_features=2)
    def forward(self, x):
        x = torch.sigmoid(self.fc1(x)) 
        x = self.output(x)
        x = F.softmax(x,dim=1)
        return x
    
model = ANN()

#Criterion and Optimizer declaration
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

#Model Training
epochs = 500
loss_arr = []
for i in range(epochs):
    y_hat = model.forward(X_train) 
    loss = criterion(y_hat, y_train) 
    loss_arr.append(loss)
    
    if i % 50 == 0:
        print(f'Epoch: {i} Loss: {loss}')
    
    optimizer.zero_grad() 
    loss.backward() 
    optimizer.step()
    
#Apply the model in the test-set.   
predict_out = model(X_test)
_,predict_y = torch.max(predict_out, 1)

#Model accuracy
print("The accuracy is ", accuracy_score(y_test, predict_y) )
