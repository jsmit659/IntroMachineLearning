#!/usr/bin/env python
# coding: utf-8

# In[1]:


# John Smith
# Student Id: 800776897
# ECGR 4105 - Intro to Machine Learning
# Homework 1

# github.com/jsmit659/IntroMachineLearning/blob/main/SmithJohn_800776897_HW1.py


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


housing = pd.DataFrame(pd.read_csv('Housing.csv'))
housing.head()                     


# In[4]:


m = len(housing)
m


# In[5]:


varlist = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']

def binary_map(x): 
    return x.map({'yes' : 1, 'no' : 0, 'furnished' : 1, 'semi-furnished' : 0.5, 'unfurnished' : 0 })

housing[varlist] = housing[varlist].apply(binary_map)

housing.head()


# In[6]:


def printData(X, color):
    plt.scatter(X,Y,color = color, marker = '.')
    plt.grid()
    plt.title('Scatter plot of training data')
    plt.rcParams["figure.figsize"] = (15,9)


# In[7]:


def compute_loss(X, Y, theta, penalty = 0):
    predictions = X.dot(theta)
    errors = np.subtract(predictions, Y)
    sqrErrors = np.square(errors)
    if (penalty == 0):
        J = 1 / (2 * m) * np.sum(np.square(errors))
    else:
        J = 1 / (2 * m) * (np.sum(np.square(errors)) + penalty * (np.sum(theta) - theta[0]))
    return J


# In[8]:


def gradient_descent(X, Y, X2, Y2, theta, alpha, iterations, penalty = 0):
    loss_history = np.zeros(iterations)
    loss_history2 = np.zeros(iterations)
    
    for i in range(iterations):
        prediction = X.dot(theta)
        errors = np.subtract(prediction, Y)
        sum_delta = (alpha / m) * X.transpose().dot(errors);
        
        if penalty == 0:
            theta = theta - sum_delta;
        else:
            theta = theta * (1 - alpha * (penalty/m)) - sum_delta
            
        loss_history[i] = compute_loss(X,Y,theta, penalty)
        loss_history2[i] = compute_loss(X2, Y2, theta)

    return theta, loss_history, loss_history2


# In[9]:


def gradientDes(loss_history, color):
    plt.plot(range(1, iterations + 1), loss_history, color = color)
    plt.rcParams["figure.figsize"] = (15,9)
    plt.grid()
    plt.xlabel('Number of Iterations')
    plt.title('Convergence of Gradient Descent')
    
def multiGradientDes(size, X, Y, X2, Y2, alpha = 0.01, penalty = 0):
    X0 = np.ones((len(X), 1))
    X = np.hstack((X0, X))
    
    X0 = np.ones((len(X2), 1))
    X2 = np.hstack((X0, X2))
    
    theta = np.zeros(size)
    iterations = 1500
    
    loss = compute_loss(X, Y, theta)
    theta, loss_history, loss_history2 = gradient_descent(X, Y, X2, Y2, theta, alpha, iterations, penalty)
    
    return theta, loss_history, loss_history2


# In[10]:


raw_data = pd.DataFrame(pd.read_csv("Housing.csv"))

m = len(raw_data)
raw_data[varlist] = raw_data[varlist].apply(binary_map)


# In[11]:


from sklearn.model_selection import train_test_split
np.random.seed(0)
train_data, test_data = train_test_split(housing, train_size = 0.8, test_size = 0.2, random_state = 42)


# In[12]:


import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def Sep_Nor(reqVars, data, Nor = 'False'):
    data = data[reqVars]
    
    if Nor == 'True':
        scaler = MinMaxScaler()
        data[reqVars] = scaler.fit_transform(data[reqVars])
        
    if Nor == 'Other':
        scaler = StandardScaler()
        data[reqVars] = scaler.fit_transform(data[reqVars])
        
    YData = data.pop('price')
    XData = data
    
    return XData, YData

def graphEverything(reqVars, Norm, alpha = 0.01, penalty = 0):
    X_train, Y_train = Sep_Nor(reqVars, train_data, Norm)
    X_test, Y_test = Sep_Nor(reqVars, test_data, Norm)

    theta_Train, cost_history_Train, cost_history_Test = multiGradientDes(len(reqVars), X_train, Y_train, X_test, Y_test, alpha, penalty)
    
    plt.rcParams["figure.figsize"] = (15,9)
    plt.grid()
    gradientDes(cost_history_Train, 'green')
    gradientDes(cost_history_Test, 'blue')
    
    return theta_Train


# In[13]:


reqVars_1 = ['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'parking']
reqVars_2 = ['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea']
iterations = 1500


# In[14]:


theta_Train = graphEverything(reqVars_1, 'False', 0.0000000003)
plt.title('Question 1: Problem A')
print('\033[0m'+'Question 1: Problem A\n')
print('The graph is unreadable unless an extremely low learning rate is used. These variables are area, bedrooms, bathrooms, stories, and parking. These are the best parameters given by theta:\n')
print('\n'.join('{}: {}'.format(*val) for val in enumerate(theta_Train)))


# In[15]:


theta_Train = graphEverything(reqVars_2, 'False', 0.0000000003)
plt.title('Question 1: Problem B')
plt.grid()
print('\033[0m'+'Question 1: Problem B\n')
print('These are the best parameters given by theta:\n')
print('\n'.join('{}: {}'.format(*val) for val in enumerate(graphEverything(reqVars_1, 'False', 0.0000000003))))


# In[16]:


print('\033[0m'+'Question 2: Problem A\n')
print('Repeat problem 1 a, this time with input normalization and input standardization as part of your pre-processing logic. \nYou need to perform two separate trainings for standardization and normalization.\n')

print('This training was significanly better with less error. 1A settled around 1-1.1 where 2A settles around 0.005; \na dramatic difference.')
# Graph with input Normalization
plt.title('Question 1: Problem B')
theta_Train = graphEverything(reqVars_1, 'True')


# In[17]:


# Graph with input standardization
plt.title('Question 2: Problem A')
theta_Train = graphEverything(reqVars_1, 'Other')


# In[18]:


theta_Train = graphEverything(reqVars_2, 'True')
print('\033[0m'+'Question 2: Problem B\n')
plt.title('Question 2: Problem B')
print('Repeat problem 1B, this time with input normalization and input standardization as part of your pre-processing logic. \nYou need to perform two separate trainings for standardization and normalization.\n')
print('This training was significanly better with less error. 1B settled around 0.1 where 2B settles around 0.005; \na dramatic difference.')


# In[19]:


plt.title('Question 2: Problem B')
theta_Train = graphEverything(reqVars_2, 'Other')
#This is the graph with the input standardization


# In[20]:


theta_Train = graphEverything(reqVars_1, 'True', 0.01, 3)
print('\033[0m'+'Question 3: Problem A\n')
plt.title('Question 3: Problem A')
print('The graph does not improve significantly.\n')
print('This training uses a penalty in its evaluation.\n')


# In[21]:


theta_Train = graphEverything(reqVars_2, 'True', 0.01, 3)
print('\033[0m'+'Question 3: Problem B\n')
plt.title('Question 3: Problem B')
print('The graph does not improve significantly.\n')
print('This training uses a penalty in its evaluation.\n')


# In[ ]:




