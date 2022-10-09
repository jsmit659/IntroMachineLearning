#!/usr/bin/env python
# coding: utf-8

# In[1]:


# HW 2
#
#
#
#


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB


# In[3]:


def matrix_print(cnf_matrix):
    class_names = [0,1]
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    sns.heatmap(pd.DataFrame(cnf_matrix), annot = True, cmap = "YlGnBu", fmt = 'g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion Matrix', y = 1.1)
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')


# In[ ]:





# In[4]:


from sklearn import metrics 

def metrics_print(y_pred):
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Precision:", metrics.precision_score(y_test, y_pred))
    print("Recall:", metrics.recall_score(y_test, y_pred))

dataset = pd.read_csv('diabetes.csv')
dataset


# In[5]:


varlist = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
dataset_x = dataset[varlist]
dataset_y = dataset['Outcome']


# In[6]:



x_train, x_test, y_train, y_test = train_test_split(dataset_x, dataset_y, train_size = 0.8, test_size = 0.20, random_state = np.random)

from sklearn.preprocessing import StandardScaler
standardscalar_x = StandardScaler()
x_train = standardscalar_x.fit_transform(x_train)
x_test = standardscalar_x.fit_transform(x_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(x_train, y_train)


# In[7]:


#Problem 1: 

from sklearn.linear_model import LogisticRegression

LR_classifier = LogisticRegression()
LR_classifier.fit(x_train, y_train)


# In[8]:


#Prints out the predictions from the Logistic Regression Model
LR_y_pred = LR_classifier.predict(x_test)
print("Y Prediction: \n\n", LR_y_pred)


# In[9]:


LR_matrix = confusion_matrix(y_test, LR_y_pred)
print("LR_Matrix: \n\n", LR_matrix)


# In[10]:


print("Logisitc Regression Metrics: \n")
metrics_print(LR_y_pred)


# In[11]:


# Confusion Matrix for Problem #1
matrix_print(LR_matrix)


# In[12]:


# Problem #2

NB_classifier = GaussianNB()
NB_classifier.fit(x_train, y_train)


# In[13]:


NB_y_prediction = NB_classifier.predict(x_test)
print("Y Predition: \n\n", NB_y_prediction)


# In[14]:


# Create the Naive Bayes Confusion Matrix

NB_matrix = confusion_matrix(y_test, NB_y_prediction)
print("NB_Matrix: \n\n", NB_matrix)


# In[15]:


# Print Metrics for Problem #2 

print("Naive Bayes Metrics: \n")
metrics_print(NB_y_prediction)


# In[16]:


matrix_print(NB_matrix)


# In[17]:


K5_scores = cross_val_score(LR_classifier, x_train, y_train, cv = 5)
K10_scores = cross_val_score(LR_classifier, x_train, y_train, cv = 10)

print("Five K-folds Scores: \n\n", K5_scores)
print("\nTen K-folds Scores: \n\n", K10_scores)


# In[18]:


# Problem 3:

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.datasets import load_breast_cancer 

breast = load_breast_cancer()
breast_data = breast.data
breast_data.shape


# In[19]:


breast_input = pd.DataFrame(breast_data)
breast_input.head() 


# In[20]:


breast_labels = breast.target 
breast_labels.shape


# In[21]:


labels = np.reshape(breast_labels, (569,1))
final_breast_data = np.concatenate([breast_data, labels], axis = 1)
final_breast_data.shape


# In[22]:


breast_dataset = pd.DataFrame(final_breast_data)
features = breast.feature_names
features_labels = np.append(features, 'label')
breast_dataset.columns = features_labels
breast_dataset


# In[23]:


breast_dataset['label'].replace(0, 'Benign',inplace=True) 
breast_dataset['label'].replace(1, 'Malignant',inplace=True) 


# In[24]:


x_train, x_test, y_train, y_test = train_test_split(dataset_x, dataset_y, train_size = 0.8, test_size = 0.20, random_state = np.random)

from sklearn.preprocessing import StandardScaler
standardscalar_x = StandardScaler()
x_train = standardscalar_x.fit_transform(x_train)
x_test = standardscalar_x.fit_transform(x_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(x_train, y_train)


# In[25]:


from sklearn.linear_model import LogisticRegression

LR_classifier = LogisticRegression()
LR_classifier.fit(x_train, y_train)


# In[26]:


#Prints out the predictions from the Logistic Regression Model
LR_y_pred = LR_classifier.predict(x_test)
print("Y Prediction: \n\n", LR_y_pred)


# In[27]:


LR_matrix = confusion_matrix(y_test, LR_y_pred)
print("LR_Matrix: \n\n", LR_matrix)


# In[28]:


print("Logisitc Regression Metrics: \n")
metrics_print(LR_y_pred)


# In[29]:


# Confusion Matrix for Problem #3
matrix_print(LR_matrix)


# In[30]:


# Problem 4

K5_scores = cross_val_score(LR_classifier, x_train, y_train, cv = 5)
K10_scores = cross_val_score(LR_classifier, x_train, y_train, cv = 10)

print("Five K-folds Scores: \n\n", K5_scores)
print("\nTen K-folds Scores: \n\n", K10_scores)


# In[ ]:


# The k-fold cross validation seem to be more accurate than in Problem #3

