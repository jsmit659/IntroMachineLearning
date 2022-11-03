#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# HW 4
# https://github.com/jsmit659/IntroMachineLearning/blob/main
# /SmithJohn_800776897_HW4.py
# John Smith
# 800776897
#


# In[1]:


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
from sklearn.decomposition import PCA
from sklearn.datasets import load_breast_cancer
# Importing required libraries
from seaborn import load_dataset, pairplot
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import max_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.metrics import r2_score


# In[2]:


def metrics_print(y_pred, y_test):
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Precision:", metrics.precision_score(y_test, y_pred))
    print("Recall:", metrics.recall_score(y_test, y_pred))

# Prints the matrix
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
    
def create_pca(scled_x, columns, n):
    
    pca = PCA(n_components = n)
    principalComponents = pca.fit_transform(scled_x) 
    principalDf = pd.DataFrame(data = principalComponents, columns = columns) 
    
    finalDf = pd.concat([principalDf, raw_y], axis = 1)
    return finalDf
    
    
def logistic_regression(raw_x, raw_y):
    # Splits the data
    x_train, x_test, y_train, y_test = train_test_split(raw_x, breast.target, test_size = 0.20, random_state = 5)
    
    # Creates model for Logistic Regression
    classifier = LogisticRegression()
    classifier.fit(x_train, y_train)

    y_pred = classifier.predict(x_test)
    metrics_print(y_pred, y_test)
    
    # Concusion Matrix
    matrix = confusion_matrix(y_test, y_pred)
    print("Matrix: \n\n", matrix)
    
    return classifier, matrix


def graph_pca(data):
    fig = plt.figure(figsize = (12,12)) 
    ax = fig.add_subplot(1,1,1)  
    ax.set_xlabel('Principal Component 1', fontsize = 15) 
    ax.set_ylabel('Principal Component 2', fontsize = 15) 
    ax.set_title('2 component PCA', fontsize = 20) 
    targets = ['Malignant','Benign']
    colors = ['r', 'g', 'b'] 
    for target, color in zip(targets,colors): 
        indicesToKeep = pca_y == target 
        ax.scatter(data.loc[indicesToKeep, 'Principal Component 1'], data.loc[indicesToKeep, 'Principal Component 2'], c = color, s = 50) 
    ax.legend(targets) 
    ax.grid()

def plot_svc_decision_function(model, ax=None, plot_support=True):
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y,x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    ax.contour(X, Y, P, colors='k', 
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--','-','--'])
    if plot_support:
        ax.scatter(model.support_[:, 0], model.support_[:, 1],
                  s=300, linewidth = 1, facecolors='none');
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    plt.scatter(X[:,0],X[:,1], c=y, s=50, cmap = 'autumn')


# In[3]:


breast = load_breast_cancer()
breast_data = breast.data
breast_data.shape

breast_input = pd.DataFrame(breast_data)
breast_input.head()

breast_labels = breast.target 
breast_labels.shape

breast_labels = np.reshape(breast_labels,(breast_labels.size,1))
final_breast_data = np.concatenate([breast_data,breast_labels],axis=1)
final_breast_data.shape

breast_dataset = pd.DataFrame(final_breast_data)
features_x = breast.feature_names
features_labels = np.append(features_x,'label')
breast_dataset.columns = features_labels

breast_dataset.head()


# In[4]:


df = breast_dataset
df.head()


# In[5]:


raw_x = breast_dataset[features_x]
raw_y = breast_dataset['label']


# In[6]:


sc_x = StandardScaler()
scled_x = sc_x.fit_transform(raw_x)

accuracylist = list()
recalllist = list()
precisionlist = list()

for n in range(1,31):
    pca = PCA(n)
    principalComponents = pca.fit_transform(scled_x)
    principalDf = pd.DataFrame(data = principalComponents)
    finalDf = pd.concat([principalDf, df[['label']]], axis = 1)
    x_train, x_test, y_train, y_test = train_test_split(principalDf, breast.target, test_size = 0.20, random_state = 12)
    classifier = SVC(kernel='rbf', C=10, random_state = 12)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    accuracylist.append(metrics.accuracy_score(y_test, y_pred))
    precisionlist.append(metrics.precision_score(y_test, y_pred))
    recalllist.append( metrics.recall_score(y_test, y_pred))


# In[7]:


plt.plot(accuracylist)
plt.plot(precisionlist)
plt.plot(recalllist)
plt.grid()
plt.title("SVM")
plt.show()


# In[8]:


n = 7;
pca = PCA(n)
principalComponents = pca.fit_transform(scled_x)
principalDf = pd.DataFrame(data = principalComponents)
finalDf = pd.concat([principalDf, df[['label']]], axis = 1)
x_train, x_test, y_train, y_test = train_test_split(principalDf, breast.target, test_size = 0.20, random_state = 12)
classifier = SVC(kernel='rbf', C=10, random_state = 12)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)


# In[23]:


print('Accuracy:  ', accuracy)
print('Precision: ', precision)
print('Recall:    ', recall)

# Question 1 Q4: Due to the use of a non-linear kernal, 
# the precision and accuracy are much
# much better than those of Homework 3


# In[24]:


# Problem 2

housing = pd.DataFrame(pd.read_csv('Housing.csv'))

varlist = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']
def binary_map(x):
    return x.map({'yes' : 1, 'no' : 0, 'furnished' : 1, 'semi-furnished' : 0.5, 'unfurnished' : 0 })
housing[varlist] = housing[varlist].apply(binary_map)

housing.head()


# In[25]:


del housing['furnishingstatus']


# In[26]:


housing.head()


# In[27]:


features = ['area','bedrooms','bathrooms', 'stories', 'mainroad', 'guestroom', 'basement', 'hotwaterheating','airconditioning','parking','prefarea']
housing = StandardScaler().fit_transform(housing)
housing=pd.DataFrame(housing)
y = housing.pop(0)
x = housing


# In[28]:


test_r2 = list()
training_r2 = list()
test_MAE = list()
training_MAE = list()
training_MSE = list()
test_MSE = list()
Max_Error = list()
def svr_select(kernel='rbf',c=10):
    for n in range(1,12):
        pca = PCA(n)
        principalComponents = pca.fit_transform(x)
        principalDf = pd.DataFrame(data = principalComponents)
        x_train, x_test, y_train, y_test = train_test_split(principalDf, y, test_size = 0.20,)
        classifier = SVR(kernel='rbf', C=10)
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
    
        test_r2.append(r2_score(y_test, y_pred))
        training_r2.append(classifier.score(x_train, y_train))

        test_MAE.append(mean_absolute_error(y_test, y_pred))
        training_MAE.append(mean_absolute_error(y_train, classifier.predict(x_train)))

        training_MSE.append(mean_squared_error(y_test, y_pred))
        test_MSE.append(mean_squared_error(y_train, classifier.predict(x_train)))
            
        Max_Error.append(max_error(y_test, y_pred))

                
def svr_graph(kernel='kernel'):
    
    plt.plot(test_r2,label = 'Test R2')
    plt.plot(training_r2,label = 'Training R2')

    plt.plot(test_MAE,label = 'Test MAE')
    plt.plot(training_MAE,label = 'Training MAE')

    plt.plot(training_MSE,label = 'Training MSE')
    plt.plot(test_MSE,label = 'Test MSE')


    plt.grid()
    plt.xlabel('data')
    plt.ylabel('target')
    plt.title(kernel)
    plt.legend()
    plt.show()


# In[29]:


svr_select(kernel='rbf')
svr_graph(kernel='rbf')


svr_select(kernel='linear')
svr_graph(kernel='linear')


svr_select(kernel='poly')
svr_graph(kernel='poly')


# In[33]:


n = 11
pca = PCA(n)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents)

x_train, x_test, y_train, y_test = train_test_split(principalDf, y, test_size = 0.20,)
classifier = SVR(kernel='linear', C = 10)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)    
test_r2.append(r2_score(y_test, y_pred))

training_r2 = (classifier.score(x_train, y_train))
test_MAE = (mean_absolute_error(y_test, y_pred))
training_MAE = (mean_absolute_error(y_train, classifier.predict(x_train)))
training_MSE = (mean_squared_error(y_test, y_pred))
test_MSE = (mean_squared_error(y_train, classifier.predict(x_train)))         
Max_Error = (max_error(y_test, y_pred))

print('Training R2:  ', training_r2)
print('Test MAE:     ', test_MAE)
print('Training MAE: ', training_MAE)
print('Training MSE: ', training_MSE)
print('Test_MSE:     ', test_MSE)
print('Max Error:    ', Max_Error)


# In[31]:


# Problem 2 Q2: Due to the SVC taking into account the 
# weight for each feature, these results are more accurate 
# than those in Homework1

# Problem 2 Q3: I believe in this dataset, linear is better 
# with peaks at 11.

# Problem 2 Q4: Plotting the different kernals, linear 
# seems to have the best fit with poly giving very similar
# results. 


# In[ ]:




