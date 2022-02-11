#!/usr/bin/env python
# coding: utf-8

# ## Justyna Komorowska

# **Credit Card Fraud Detection**
# 
# 
# Dataset: creditcard.csv data is anonymized credit card transactions labeled as fraudulent or genuine. `Class` is the target variable and stands for 0 if there was no fraud and 1 if contrary.
# 
# Tasks:\
# •	Exploratory Data Analysis\
# •	Classification with several ML methods\
# •	Model Selection.

# **Exploratory Data Analysis**
# 
# Train-test split\
# Data Visualization\
# Oversampling
# 
# **Modelling**
# 
# Classification based on following algorithms:
# 
# 1. Logistic Regression
# 2. Naive Bayes Classifier
# 3. K-NN
# 4. Decision Tree
# 5. Neural Network (2 layers)
# 
# **Models performance comparison**
# 1. ROC curve
# 2. Metrics extraction: Precision, Sensitivity, Accuracy, AUC.
# 

# In[3]:


import pandas as pd
import math
import numpy as np

#data visualisation
import seaborn as sns
import matplotlib.pyplot as plt
from plotnine import *

#preprocessing
from sklearn.utils import resample
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split

#modeling
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from tensorflow import keras
from sklearn.svm import SVC


#Precision, Sensitivity, Accuracy, AUC
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc

import pickle 
import warnings
warnings.filterwarnings('ignore')


# In[4]:


PATH = 'creditcard.csv'
dane = pd.read_csv(PATH, sep = ',')
dane.head()


# In[5]:


dane.info()


# In[6]:


#No nulls
dane.isna().sum().sum()


# In[7]:


round(dane.describe(), 4)


# #### Data summary:<BR>
#     
# Preliminary distribution analysis show taht variables  V1..V28 are standarized (average equal zero) and `Time`, `Amount` and target variable `Class` were not standarized.
# 
# Average value of `Amount` variable is equal 88 units (assuming dollars) and based on histogram below the distribution is strongly skewed to the right.   

# In[7]:


sns.distplot(dane['Amount'], hist=True, kde=True, 
             bins=int(1000), color = 'blue')


# In[10]:


plt.hist(dane.Time)
plt.title('Histogram dla zmiennej Time')

#Bimodal distribution


# In[11]:


#Additional check if the target variable correlates with Time variable - nope
(ggplot(dane, aes('Time'))
 + geom_histogram()
 + facet_wrap('~Class'))


# For `Class` variable values equal 1 we can see lower values of the `Amount` value. Based on the visualization below one could propose a simple heuristic that for amounts higher then X we do not expect fraudulent transactions.

# In[12]:


sns.relplot(x="Class", y="Amount", hue='Class', data=dane)


# ### Class Imbalance

# In[9]:


sns.countplot('Class', data=dane)
plt.title('Class Distributions \n (0: Klasa 0 || 1: Klasa 1)', fontsize=14)


# In[13]:


pd.crosstab(index=dane['Class'], columns='count')


# In[16]:


492/28315 #


# In this dataset we observe very high class imbalance. Ratio is 99.83% to 0.17%. Using the raw data set as-is introduces a risk of overtraining the resulting model i.e. with such a high class imbalance there is a risk that during model trainig we would end up in a local optimum with the model only predicts one class for all observations. For low amount of observation with `Class` equal 1 it's hard to observe correlations between values.
# 
# There are at least 4 options we can consider:
# 1. Adding weights for observations from minority class.
# 2. Oversampling - multiplicating of instances of minority class.
# 3. Undersampling - removing of of instances of majority.

# #### Back-up

# In[9]:


df = dane.copy()
df.shape


# ####  `Time` and `Amount` scaling

# In[10]:


# RobustScaler is less prone to outliers.
std_scaler = StandardScaler()
rob_scaler = RobustScaler()

df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1))
df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1,1))
df.drop(['Time','Amount'], axis=1, inplace=True)


# #### Train test split

# In[11]:


X = df # 'Class' not removed - will be needed for EPA step.
y = df['Class']

# train_test_split
X_train_EDA, X_test1, y_beta, y_test = train_test_split(X, y, test_size = 0.25, stratify = y, random_state = 0)
X_test = X_test1.drop(['Class'], axis = 1)

#'Class' variable is left in X_train_EDA sub-set for oversampling and exploratory data analysis steps.
# y_beta is not going to be used. y_train will be created with "Class" column from X_train_EDA sub-set once data analysis
# will be done.


# In[12]:


X_test.shape


# In[13]:


y_test.shape


# In[15]:


X_train_EDA.shape


# In[16]:


pd.crosstab(index=X_train_EDA['Class'], columns='count')


# ### Data Manipulation on train sub-set

# #### Oversampling

# In[23]:


# Minority and majority observations separation

df_majority = X_train_EDA[X_train_EDA['Class']==0]
df_minority = X_train_EDA[X_train_EDA['Class']==1]

# minority class oversampling
# n_samples value is chosen to create a 50/50 sub-dataframe ratio of "Fraud" and "Non-Fraud" transactions.

df_minority_os = resample(df_minority, replace = True, n_samples = 213236, random_state = 0)
df_overs = pd.concat([df_majority, df_minority_os])
df_overs.shape


# In[33]:


sns.countplot('Class', data=df_overs)
plt.title('Class Distributions \n (0: Klasa 0 || 1: Klasa 1)', fontsize=14)


# In[35]:


fig, ax = plt.subplots(figsize=(20,15)) 
sns.heatmap(round(X_train_EDA.corr(), 2), annot = True, ax = ax)
plt.title("Correlogram Raw but Scaled Data", fontsize = 40)


# In[36]:


fig, ax = plt.subplots(figsize=(20,15)) 
sns.heatmap(round(df_overs.corr(), 2), annot = True, ax = ax)
plt.title("Correlogram Oversampled and Scaled Data", fontsize = 40)


# **Correlogram conclusions:**<BR>
#     
#     Heatmap reveals strong linear correlations for `V16`, `V17` and `V18`. These three will be monitored. Possible steps:
#     1. VIF calculation
#     2. removing 
#     3. some arithmetical transformation
#    

# ## Exploratory Data Analysis for train subset

# Based on the below for modelling I'm choosing only the variables that shows separation in the `Class` variable values.<BR>
#     **Variables that stay in**: <BR>
#     `V1`, `V2`, `V3`, `V4`, `V5`, `V6`, `V7`, `V8`, `V9`, `V10`, `V11`, `V12`, `V14`, `V16`, `V17`, `V18`<BR>
#     **The following variables will not be considered:**<BR>
#     `V13`, `V15`, `V19`, `V20`, `V22`, `V23`, `V24`, `V25`, `V26`, `V28`, `scaled_time`, `scaled_amount` <BR>
# 
# `V9` and `V10` seem to be a pair of good candidates for dimensionality reduction.

# In[37]:


cols = df_overs.columns

for i in range(len(cols)):
    
    fig, ax = plt.subplots(figsize= (12,8))
    sns.histplot(x=cols[i], hue = "Class", alpha = 0.3, data = df_overs)
    plt.xlabel(cols[i])
    plt.show()


# In[39]:


#Additional zoomed-in histogram for Scaled Data
fig, ax = plt.subplots(figsize= (12,8))
sns.histplot(x="scaled_amount",
                hue='Class', binwidth = 1, data = df_overs, alpha = 0.3)


# In[41]:


cols = df_overs.columns

for i in range(len(cols)):
    for j in cols[i+1:]:
        fig, ax = plt.subplots(figsize= (12,8))
        sns.scatterplot(x=cols[i], y=j, hue = "Class", alpha = 0.3, data = df_overs)
        plt.xlabel(cols[i])
        plt.ylabel(j)
        plt.show()


# ####  Final train and test subsets preparation

# In[18]:


#Removing variables that explain no target value variablility.
columns_drop = ['V13', 'V15', 'V19', 'V20', 'V22', 'V23', 'V24', 'V25', 'V26', 'V28','scaled_time','scaled_amount']
X_test.drop(columns_drop, inplace=True, axis=1) 


# In[19]:


X_test.head()


# In[24]:


# X_train and y_train sets preparation: removing Class and unwanted columns.

y_train = df_overs["Class"]
columns_drop_train = ['V13', 'V15', 'V19', 'V20', 'V22', 'V23', 'V24', 'V25', 'V26', 'V28','scaled_time','scaled_amount', 'Class']
df_overs.drop(columns_drop_train, inplace=True, axis=1)


# In[25]:


X_train = df_overs # name change for sake of convention for further steps 


# Heatmap reveals strong linear correlations for `V11`, `V12` and `V14`. If there was a regression problem - there is additional wrangling needed. For Classification problem I'll risk leaving the data as-it-is and monitoring. 

# In[110]:


fig, ax = plt.subplots(figsize=(20,15)) 
sns.heatmap(round(X_train.corr(), 2), annot = True, ax = ax)
plt.title("Correlogram X_train", fontsize = 40)


# # Modelling

# Models will be trained on oversampled dataset but tested on data with original class imbalance. I'm assuming that this is a real data structure and I want my model to perform on real data.

# ## Logistic Regression

# In[26]:


#Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train,y_train)
y_pred_logreg = logreg.predict(X_test) #array([0,0,..,1])
p_pred_logreg = logreg.predict_proba(X_test) # Returns probability estimates for the test vector X.


# In[27]:


#Metrics
accuracy = accuracy_score(y_test, y_pred_logreg)
print(f"Accuracy : {accuracy}")
precision = precision_score(y_test, y_pred_logreg)
print(f"Precision : {precision}")
recall = recall_score(y_test, y_pred_logreg)
print(f"Recall / TPR : {recall}")
AUC = roc_auc_score(y_test, y_pred_logreg)
print(f"Area Under Curve : {AUC}")
print(" "*3)
cnf_matrix = confusion_matrix(y_test, y_pred_logreg) 
print('Confusionn matrix for Logistic Regression:\n', cnf_matrix)


# ## Naive Bayes Classifier

# In[29]:


# Naive Bayes Classifier
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_NBC = gnb.predict(X_test)
p_pred_NBC = gnb.predict_proba(X_test) #Returns probability estimates for the test vector X.


# In[30]:


accuracy = accuracy_score(y_test, y_pred_NBC)
print(f"Accuracy : {accuracy}")
precision = precision_score(y_test, y_pred_NBC)
print(f"Precision : {precision}")
recall = recall_score(y_test, y_pred_NBC)
print(f"Recall / TPR : {recall}")
AUC = roc_auc_score(y_test, y_pred_NBC)
print(f"Area Under Curve : {AUC}")
print(" " *3)
cnf_matrix = confusion_matrix(y_test, y_pred_NBC) 
print('Confusionn matrix for Naive Bayes Classifier:\n', cnf_matrix)


# ## KNN (K-Nearest Neighbors)

# In[33]:


#KNN test for K = sqrt(len(df)) and Minkowski distance
math.sqrt(X_train.shape[0])


# In[34]:


KNN = KNeighborsClassifier(n_neighbors = 653)
KNN.fit(X_train, y_train)
y_predKNN = KNN.predict(X_test)
p_pred_KNN = KNN.predict_proba(X_test)


# In[35]:


accuracy = accuracy_score(y_test, y_predKNN)
print(f"Accuracy : {accuracy}")
precision = precision_score(y_test, y_predKNN)
print(f"Precision : {precision}")
recall = recall_score(y_test, y_predKNN)
print(f"Recall / TPR : {recall}")
AUC = roc_auc_score(y_test, y_predKNN)
print(f"Area Under Curve : {AUC}")
print(" " *3)
cnf_matrix = confusion_matrix(y_test, y_predKNN) 
print('Confusionn matrix for 653 Nearest Neighbours:\n', cnf_matrix)


# ## Decision Tree

# In[38]:


tree = tree.DecisionTreeClassifier(max_depth = 15)
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)
p_pred_tree = tree.predict_proba(X_test)


# In[39]:


accuracy = accuracy_score(y_test, y_pred_tree)
print(f"Accuracy : {accuracy}")
precision = precision_score(y_test, y_pred_tree)
print(f"Precision : {precision}")
recall = recall_score(y_test, y_pred_tree)
print(f"Recall / TPR : {recall}")
AUC = roc_auc_score(y_test, y_pred_tree)
print(f"Area Under Curve : {AUC}")
print(" "*3)
cnf_matrix = confusion_matrix(y_test, y_pred_tree) 
print('Confusionn matrix for unpruned Decision Tree:\n', cnf_matrix)


# ## Neural Network

# In[41]:


modelNN = keras.Sequential(
    [
        keras.layers.Dense(
            10, activation="relu", input_shape=(X_train.shape[-1],)
        ),
        keras.layers.Dense(10, activation="relu"),

        keras.layers.Dense(1, activation="sigmoid"),
    ]
)
modelNN.summary()


# In[42]:


metrics = [
#    keras.metrics.FalseNegatives(name="fn"),
#    keras.metrics.FalsePositives(name="fp"),
#    keras.metrics.TrueNegatives(name="tn"),
#    keras.metrics.TruePositives(name="tp"),
    keras.metrics.Precision(name="precision"),
    keras.metrics.Recall(name="recall"),
]

modelNN.compile(
    optimizer=keras.optimizers.Adam(1e-2), loss="binary_crossentropy", metrics=metrics
)

callbacks = [keras.callbacks.ModelCheckpoint("NN_model_at_epoch_{epoch}.h5")]

modelNN.fit(
    X_train,
    y_train,
    batch_size=2048,
    epochs=30,
    verbose=2,
    callbacks=callbacks,
    validation_data=(X_test, y_test)
)


# In[43]:


p_pred_NN = modelNN.predict(X_test) #p_pred_NN to są prawdopodobienstwa: trzeba ustawić treshold, wtedy dostanę 0 i 1.


# In[44]:


import numpy as np
# extract the predicted class labels
y_pred_NN = np.where(p_pred_NN > 0.5, 1, 0)
print(y_pred_NN)


# In[45]:


accuracy = accuracy_score(y_test, y_pred_NN)
print(f"Accuracy : {accuracy}")
precision = precision_score(y_test, y_pred_NN)
print(f"Precision : {precision}")
recall = recall_score(y_test, y_pred_NN)
print(f"Recall / TPR : {recall}")
AUC = roc_auc_score(y_test, y_pred_NN)
print(f"Area Under Curve : {AUC}")
print(" "*3)
cnf_matrix = confusion_matrix(y_test, y_pred_NN) 
print('Confusionn matrix for Neural Network:\n', cnf_matrix)


# # ROC Comparison

# In[156]:


#Logit
logreg_fpr, logreg_tpr, threshold = roc_curve(y_test, p_pred_logreg[:, 1])
auc_logreg = auc(logreg_fpr, logreg_tpr)

#Naive Bayesa
NBC_fpr, NBC_tpr, threshold = roc_curve(y_test, p_pred_NBC[:, 1])
auc_NBC = auc(NBC_fpr, NBC_tpr)

#KNN
KNN_fpr, KNN_tpr, threshold = roc_curve(y_test, p_pred_KNN[:, 1])
auc_KNN = auc(KNN_fpr, KNN_tpr)

#Decision Tree
tree_fpr, tree_tpr, threshold = roc_curve(y_test, p_pred_tree[:, 1])
auc_tree = auc(tree_fpr, tree_tpr)

#Neural Network 
NN_fpr, NN_tpr, threshold = roc_curve(y_test, p_pred_NN)
auc_NN = auc(NN_fpr, NN_tpr)



plt.figure(figsize=(10, 10), dpi=100)
plt.plot(logreg_fpr, logreg_tpr, label='Logistic (AUC = %0.3f)' % auc_logreg)
plt.plot(NBC_fpr, NBC_tpr, label='Naive Bayes Classifier (AUC = %0.3f)' % auc_NBC)
plt.plot(KNN_fpr, KNN_tpr, label='KNearestNeighbors (AUC = %0.3f)' % auc_KNN)
plt.plot(tree_fpr, tree_tpr, label='Decision Tree (AUC = %0.3f)' % auc_tree)
plt.plot(NN_fpr, NN_tpr, linestyle='-', label='NeuralNetwork (AUC = %0.3f)' % auc_NN)



plt.xlabel('False Positive Rate -->')
plt.ylabel('True Positive Rate -->')

plt.legend()

plt.show()


# ROC comparison:<BR>
# On basis of Area Under Curve - Logistic Regression has the highest value.
# There is no threshold given so my recommendation would be Logistic Regression.

# My Lesson Learned:<BR>
# 
# What I have learned during this project is that Oversampling step one need to do **after** train_test_split step. If these two steps are reversed, duplicated observations are divided between train and test sub-sets. 

# Studied articles:<BR>
# https://www.kaggle.com/mlg-ulb/creditcardfraud/discussion/277570

# In[ ]:




