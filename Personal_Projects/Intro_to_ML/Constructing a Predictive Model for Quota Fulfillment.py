# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 12:56:43 2022

@author: tanne
"""
###############################################################################

## Direct imports
#import io
#import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

## From imports
#from scipy import stats
#from collections import Counter

# pip install -U imbalanced-learn
from imblearn.over_sampling import SMOTE 
from imblearn.over_sampling import ADASYN

# pip install tensorflow
from keras.layers import Dense
from keras.models import Sequential

from sklearn import preprocessing
#from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

###############################################################################




################################## Functions ##################################

### Declaring initial lists for metric and model ID storage
model = list()
resample = list()
precision = list()
recall = list()
F1score = list()
AUCROC = list()

def test_eval(clf_model, X_test, y_test, algo=None, sampling=None):
    # Test set prediction
    y_prob = clf_model.predict_proba(X_test)
    y_pred = clf_model.predict(X_test)

    print('Confusion Matrix')
    print('='*60)
    print(confusion_matrix(y_test,y_pred),"\n")
    print('Classification Report')
    print('='*60)
    print(classification_report(y_test,y_pred),"\n")
    print('AUC-ROC')
    print('='*60)
    print(roc_auc_score(y_test, y_prob[:,1]))
          
    model.append(algo)
    precision.append(precision_score(y_test,y_pred))
    recall.append(recall_score(y_test,y_pred))
    F1score.append(f1_score(y_test,y_pred))
    AUCROC.append(roc_auc_score(y_test, y_prob[:,1]))
    resample.append(sampling)

def Baseline_Logit(trainX, trainY, testX, testY, keyword):
    log_model=LogisticRegression(max_iter = 10000)
    params={'C':np.linspace(.01,10,80)}
    cv = StratifiedKFold(n_splits=5, random_state=100, shuffle=True)
    
    clf_LR = GridSearchCV(log_model, params, cv=cv, scoring='roc_auc', n_jobs=-1)
    clf_LR.fit(trainX, trainY)
    
    optimal_estimator = clf_LR.best_estimator_
    predictions = clf_LR.predict(testX)
    
    test_eval(clf_LR, testX, testY, 'Logistic Regression', 'actual')

    return clf_LR

def Baseline_KNN(trainX, trainY, testX, testY, neighbors, keyword):
    
    ### Iterating training and testing scores over range of neighbors input
    neighbors_arr = np.arange(1, neighbors + 1)
    train_accuracy = np.zeros(neighbors)
    test_accuracy = np.zeros(neighbors)

    for i in neighbors_arr:
        knn = KNeighborsClassifier(n_neighbors = i)
        knn.fit(trainX, trainY)
        predictions = knn.predict(testX)
        predictions_binary = np.round(predictions)
            
        # Compute accuracy on the training set
        predictions = knn.predict(trainX)
        predictions_binary = np.round(predictions)
        train_accuracy[i-1] = knn.score(trainX, trainY)

        # Compute accuracy on the testing set
        predictions = knn.predict(testX)
        test_accuracy[i-1] = knn.score(testX, testY)
            
    # Visualization of k values vs accuracy
    plt.figure(figsize = (14,8))
    plt.title(f'{keyword} k-NN: Varying Number of Neighbors')
    plt.plot(neighbors_arr, test_accuracy, label = 'Testing Accuracy')
    plt.plot(neighbors_arr, train_accuracy, label = 'Training Accuracy')
    plt.grid()
    plt.legend()
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Accuracy')
        
    return knn, predictions, train_accuracy, test_accuracy

### For keras documentation, refer to https://keras.io/getting_started/intro_to_keras_for_engineers/
### For additional assistance refer to ...
### ...https://machinelearningmastery.com/binary-classification-tutorial-with-the-keras-deep-learning-library/

def Baseline_NN(trainX, trainY, testX, testY):
    
    # Constructing validation subsets from training samples
    trainX, valX, trainY, valY = train_test_split(trainX, trainY, test_size = 0.33, random_state = 100)
    
    # Model architecture
    model = Sequential()
    model.add(Dense(60, input_dim = 24, activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))
    
	# Compile model and record metrics for accuracy
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    
    # Recording and outputing training metrics
    model.fit(trainX, trainY, epochs = 100, batch_size = 32, validation_data = (valX,valY))
    predictions = model.predict(testX)
    
    return predictions

###############################################################################
    



######################## Data Exploration and Cleaning ########################

## Initial exploration of data
GWP = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00597/garments_worker_productivity.csv')
GWP['date'] = pd.to_datetime(GWP['date'])
# GWP.head()
# GWP.info()
# GWP.isnull().sum()

### Removing leading spaces from strings in 'department'
# GWP['department'].unique()
GWP['department'] = GWP['department'].apply(lambda x: x.strip())

### Creating productivity binary variables
### If      productivity >= targeted productivity, then productivity flag = 1, else 0
GWP['productive'] = np.where(GWP['actual_productivity'] >= GWP['targeted_productivity'] , 1, 0)
# GWP.head()

### Checking total productivy flag counts
# GWP.groupby('productive').mean()
sns.countplot(x = 'productive', data = GWP)

### Checking productivy flag counts between sewing and finishing departments
pd.crosstab(GWP.department, GWP.productive).plot(kind = 'bar')
plt.title('Productivity by Team')
plt.xlabel('Team')
plt.ylabel('Productivity')

### Checking productivy flag counts between days
pd.crosstab(GWP.day, GWP.productive).plot(kind = 'bar')
plt.title('Productivity by Day')
plt.xlabel('Day')
plt.ylabel('Productivity')

### Checking productivy flag counts between quarters
pd.crosstab(GWP.quarter, GWP.productive).plot(kind = 'bar')
plt.title('Productivity by Quarter')
plt.xlabel('Quarter')
plt.ylabel('Productivity')

### Gathering information on workers
WorkerStats = [GWP.no_of_workers.unique(), GWP.no_of_workers.max(), GWP.no_of_workers.min()]

### Creating team size object variable
conditions = [(GWP['no_of_workers'] <= 10),
    (GWP['no_of_workers'] > 10) & (GWP['no_of_workers'] <= 30),
    (GWP['no_of_workers'] > 30) & (GWP['no_of_workers'] <= 50),
    (GWP['no_of_workers'] > 50)]

values = ['Small', 'Medium', 'Large', 'Extra Large']
GWP['team_type'] = np.select(conditions, values)
GWP = GWP.drop('date', 1)
cols = GWP.columns.tolist()
cols = cols[-1:] + cols[:-1]
GWP = GWP[cols]

###############################################################################




####################### Data Manipulation/Transformation ######################

### Dropping NaNs
GWP = GWP.dropna()

### Dummies and (0,1) variable rebalancing
GWP = pd.get_dummies(GWP)
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
GWP.iloc[:,1:11] = min_max_scaler.fit_transform(GWP.iloc[:,1:11])
GWP = GWP.drop(['team','quarter_Quarter5'], 1)


### Final check on GWP
# print(GWP.info())

###############################################################################




############################### Balancing Data ################################

### "*": Implemented

### Options:
### 1) Undersampling
### 2) Oversampling
###     2a. SMOTE resampling     * 
###     2b. ADASYN resampling    *

### Declaring input features and output labels
X = GWP.drop('productive', 1)
Y = GWP["productive"]

### SMOTE resample of dataset
### "SMOTE": SMT
sm = SMOTE(random_state = 2022)
X_SMT, Y_SMT = sm.fit_resample(X, Y)
SMT_DF = pd.DataFrame(Y_SMT)

plt.figure()
sns.countplot(x = 'productive', data = SMT_DF)

### ADASYN resample of dataset
### "ADASYN": ADA
ada = ADASYN(random_state = 2022)
X_ADA, Y_ADA = ada.fit_resample(X, Y)
ADA_DF = pd.DataFrame(Y_ADA)

plt.figure()
sns.countplot(x = 'productive', data = ADA_DF)

### Constructing split for unmodified dataset
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.2, random_state = 100)

### Constructing SMOTE and ADASYN resampled splits
X_SMT_Train, X_SMT_Test, Y_SMT_Train, Y_SMT_Test = train_test_split(X_SMT, Y_SMT, test_size = 0.2, random_state = 100)
X_ADA_Train, X_ADA_Test, Y_ADA_Train, Y_ADA_Test = train_test_split(X_ADA, Y_ADA, test_size = 0.2, random_state = 100)

###############################################################################




#################### Classification - Logistic Regression #####################

### Logit classification under unmodified dataset
LR_Model = Baseline_Logit(X_Train, Y_Train, X_Test, Y_Test, 'Unmod')

### Logit classification under SMOTE resampling
LR_Model_SMT = Baseline_Logit(X_SMT_Train, Y_SMT_Train, X_SMT_Test, Y_SMT_Test, "SMOTE")

### Logit classification under ADASYN resampling
LR_Model_ADA = Baseline_Logit(X_ADA_Train, Y_ADA_Train, X_ADA_Test, Y_ADA_Test, 'ADASYN')

###############################################################################




############################ Classification - KNN #############################

### KNN classification for unmodified dataset
### Optimal k approx.
Baseline_KNN(X_Train, Y_Train, X_Test, Y_Test, 100, "Unmod")

### KNN classification under SMOTE resampling
### Optimal k approx. 17
Baseline_KNN(X_SMT_Train, Y_SMT_Train, X_SMT_Test, Y_SMT_Test, 100, "SMOTE")

### KNN classification under ADASYN resampling
### Optimal k approx. 20
Baseline_KNN(X_ADA_Train, Y_ADA_Train, X_ADA_Test, Y_ADA_Test, 100, "ADASYN")

###############################################################################




####################### Classification - Neural Network #######################

### NN classification for unmod. dataset
NN_Predict = Baseline_NN(X_Train, Y_Train, X_Test, Y_Test)
NN_PredictBinary = np.round(NN_Predict)

### NN classification under SMOTE resampling
SMT_NN_Predict = Baseline_NN(X_SMT_Train, Y_SMT_Train, X_SMT_Test, Y_SMT_Test)
SMT_NN_PredictBinary = np.round(SMT_NN_Predict)

### NN classification under ADASYN resampling
ADA_NN_Predict = Baseline_NN(X_ADA_Train, Y_ADA_Train, X_ADA_Test, Y_ADA_Test)
ADA_NN_PredictBinary = np.round(ADA_NN_Predict)

### Computing and visualizing the confusion matrices for SMOTE and ADASYN resampling
cf_matrix = confusion_matrix(Y_Test, NN_PredictBinary)
SMT_cf_matrix = confusion_matrix(Y_SMT_Test, SMT_NN_PredictBinary)
ADA_cf_matrix = confusion_matrix(Y_ADA_Test, ADA_NN_PredictBinary)
fig, ax = plt.subplots(3, 1, figsize = (14,14))
sns.heatmap(cf_matrix, annot=True, cmap='Blues', ax = ax[0])
sns.heatmap(SMT_cf_matrix, annot=True, cmap='Blues', ax = ax[1])
sns.heatmap(ADA_cf_matrix, annot=True, cmap='Blues', ax = ax[2])

###############################################################################




############################# Classification - SVM ############################
# import SVC classifier
from sklearn.svm import SVC
# import metrics to compute accuracy
from sklearn.metrics import accuracy_score
# instantiate classifier with default hyperparameters
svc=SVC() 

# fit classifier to training set
svc.fit(X_Train,Y_Train)


# make predictions on test set
Y_Pred=svc.predict(X_Test)


# compute and print accuracy score
print('Model accuracy score with default hyperparameters: {0:0.4f}'.format(accuracy_score(Y_Test, Y_Pred)))
print('Model recall score with default hyperparameters: {0:0.4f}'.format(recall_score(Y_Test, Y_Pred)))


#Deafult with SMOTE
# fit classifier to training set
svc.fit(X_SMT_Train,Y_SMT_Train)


# make predictions on test set
Y_Pred=svc.predict(X_SMT_Test)


# compute and print accuracy score
print('Model accuracy score with default hyperparameters: {0:0.4f}'.format(accuracy_score(Y_SMT_Test, Y_Pred)))
print('Model recall score with default hyperparameters: {0:0.4f}'.format(recall_score(Y_SMT_Test, Y_Pred)))

#Default ADASYN
svc.fit(X_ADA_Train,Y_ADA_Train)


# make predictions on test set
Y_Pred=svc.predict(X_ADA_Test)


# compute and print accuracy score
print('Model accuracy score with default hyperparameters: {0:0.4f}'.format(accuracy_score(Y_ADA_Test, Y_Pred)))
print('Model recall score with default hyperparameters: {0:0.4f}'.format(recall_score(Y_ADA_Test, Y_Pred)))

###############################################################################

###Grid search unbalanced data
param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
grid.fit(X_Train,Y_Train)
print(grid.best_estimator_)
 
 
grid_predictions = grid.predict(X_Test)
print(confusion_matrix(Y_Test,grid_predictions))
print(classification_report(Y_Test,grid_predictions))
 
###SMOTE data
param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
grid.fit(X_SMT_Train,Y_SMT_Train)
print(grid.best_estimator_)
 
 
grid_predictions = grid.predict(X_SMT_Test)
print(confusion_matrix(Y_SMT_Test,grid_predictions))
print(classification_report(Y_SMT_Test,grid_predictions))
 
### ADASYN
param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
grid.fit(X_ADA_Train,Y_ADA_Train)
print(grid.best_estimator_)

grid_predictions = grid.predict(X_ADA_Test)
print(confusion_matrix(Y_ADA_Test,grid_predictions))
print(classification_report(Y_ADA_Test,grid_predictions))


