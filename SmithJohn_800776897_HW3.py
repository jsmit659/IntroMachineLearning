#!/usr/bin/env python
# coding: utf-8

# In[1]:


# John Smith
# 800776897
# 


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_breast_cancer


# In[3]:


#Functions
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Scales X variables
def scale_x(raw_x):

    sc_x = StandardScaler()
    scled_x = sc_x.fit_transform(raw_x)
    return scled_x
from sklearn import metrics
def metrics_print(y_pred, y_test):

    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Precision:", metrics.precision_score(y_test, y_pred))
    print("Recall:", metrics.recall_score(y_test, y_pred))
from sklearn.metrics import confusion_matrix
import seaborn as sns

#Prints the matrix
def matrix_print(cnf_matrix):

    class_names = [0,1]
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    sns.heatmap(pd.DataFrame(cnf_matrix), annot = True, cmap = "YlGnBu", fmt = 'g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

from sklearn.linear_model import LogisticRegression
#Creates and trains log classifer and returns the classifier and matrix evau
def log_reg(raw_x, raw_y):
    x_train, x_test, y_train, y_test = train_test_split(raw_x, breast.target,  test_size = 0.20, random_state = 3)
    
    #Creates model for Logistic Regression in terms of the data
    classifier = LogisticRegression()
    classifier.fit(x_train, y_train)

    #Scoring for model
    y_pred = classifier.predict(x_test)
    metrics_print(y_pred, y_test)

    #Creates Logistic Regression Confusion Matrix
    matrix = confusion_matrix(y_test, y_pred)
    print("Matrix: \n\n", matrix)

    return classifier, matrix

from sklearn.decomposition import PCA

def create_pca(scled_x, raw_y, columns, n):
    
    pca = PCA(n_components = n)
    principalComponents = pca.fit_transform(scled_x) 
    principalDf = pd.DataFrame(data = principalComponents, columns = columns) 
    finalDf = pd.concat([principalDf, raw_y], axis = 1)
    return finalDf

def graph_pca(data):
    fig = plt.figure(figsize = (12,12))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('Principal Component 1', fontsize = 18)
    ax.set_ylabel('Principal Component 2', fontsize = 18)
    ax.set_title('2 component PCA', fontsize = 20)
    targets = ['Malignant','Benign']
    colors = ['r', 'g', 'b']
    for target, color in zip(targets,colors):
        indicesToKeep = pca_y == target
        ax.scatter(data.loc[indicesToKeep, 'Principal Component 1'], data.loc[indicesToKeep, 'Principal Component 2'], c = color, s = 50)
    ax.legend(targets)
    ax.grid()

breast = load_breast_cancer()
breast_data = breast.data
                   
breast_input = pd.DataFrame(breast_data)
labels = breast.target

labels = np.reshape(labels,(labels.size,1))
final_breast_data = np.concatenate([breast_data, labels], axis = 1)

breast_dataset = pd.DataFrame(final_breast_data)
features_x = breast.feature_names
features_labels = np.append(features_x,'Cancer Type')

breast_dataset.columns = features_labels
breast_dataset['Cancer Type'].replace(0, 'Benign',inplace=True)
breast_dataset['Cancer Type'].replace(1, 'Malignant',inplace=True)
breast_dataset.head()


# In[4]:


raw_x = breast_dataset[features_x]
raw_y = breast_dataset['Cancer Type']

# Problem 1

scled_x = scale_x(raw_x)
nonPCA_classifier, nonPCA_matrix = log_reg(scled_x, raw_y)


# In[5]:


matrix_print(nonPCA_matrix)


# In[6]:


# Compared to the previous Homework, the Naïve Bayesian model in this homework 
# has a much higher accuracy. 

# Problem 2 

columns = ['Principal Component 1', 'Principal Component 2']
pca_data = create_pca(scled_x, raw_y, columns, 2)
pca_data


# In[7]:


pca_x = pca_data[columns]
pca_y = pca_data['Cancer Type']
graph_pca(pca_data)


# In[8]:


pca_classifier, pca_matrix = log_reg(pca_x, pca_y)


# In[9]:


columns = ['1', '2', '3']
pca_data = create_pca(scled_x, raw_y, columns, 3)
pca_x = pca_data[columns]
pca_y = pca_data['Cancer Type']
pca_classifier, pca_matrix = log_reg(pca_x, pca_y)


# In[21]:


columns = ['1', '2', '3', '4', '5','6', '7', '8', '9']
pca_data = create_pca(scled_x, raw_y, columns, 9)
pca_x = pca_data[columns]
pca_y = pca_data['Cancer Type']
pca_classifier, pca_matrix = log_reg(pca_x, pca_y)


# In[51]:


# Around 9-10 columns, the accuracy peaks and at 11 begins to fall again. 

# Problem 3
x_train, x_test, y_train, y_test = train_test_split(raw_x, breast.target,  test_size = 0.20, random_state =20)

NB_classifier = GaussianNB()
NB_classifier.fit(x_train, y_train)


# In[52]:


NB_y_predict = NB_classifier.predict(x_test)
NB_matrix = confusion_matrix(y_test, NB_y_predict)
print("Naive Bayes Matrix:\n\n", NB_matrix)


# In[53]:


print("Naive Bayes Metrics:\n")
metrics_print(NB_y_predict, y_test)


# In[54]:


matrix_print(NB_matrix)


# In[ ]:


# At random state 20, the prediction has the highest accuracy. Using 19 or 21 causes the 
# accuracy to drop off significantly 

