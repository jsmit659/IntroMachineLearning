#!/usr/bin/env python
# coding: utf-8

# In[916]:


#John Smith
#Student Id: 800776897
#ECGR 4105 - Intro to Machine Learning
#Homework 0


# In[917]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 


# In[918]:


df = pd.read_csv('D3.csv')
df.head()
M = len(df)


# In[919]:


len(df)


# In[920]:


dataset = df.values[:,:]
print('dataset = ', dataset[:10,:])


# In[942]:


X= df.values[:,0]
K= df.values[:,1]
Z= df.values[:,2]
Y= df.values[:,3]

m = len(j)

X_0 = X
X_1 = K
X_2 = Z


# In[922]:


def printData(X, color):
    plt.scatter(X,Y,color = color)
    plt.grid()
    plt.title('Scatter plot of training data')
    plt.rcParams["figure.figsize"] = (10,6)
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')


# In[923]:


def compute_loss(X, K, theta):
    predictions = X.dot(theta)
    errors = np.subtract(predictions, Y)
    sqrErrors = np.square(errors)
    J = 1 / (2 * m) * np.sum(np.square(errors))
    return J


# In[924]:


def gradient_descent(X, Y, theta, alpha, iterations):
    loss_history = np.zeros(iterations)
    
    for i in range(iterations):
        prediction = X.dot(theta)
        errors = np.subtract(prediction, Y)
        sum_delta = (alpha / m) * X.transpose().dot(errors);
        theta = theta - sum_delta;
        loss_history[i] = compute_loss(X,Y,theta)

    return theta, loss_history


# In[925]:


def graphRegression(W, T, color):
    X0 = np.ones((m,1))
    X1 = T.reshape(m,1)
    W = np.hstack((X0,X1))
    theta = np.zeros(2)
    iterations = 1500;
    alpha = 0.01;

    loss = compute_cost(W,Y,theta)
    theta, loss_history = gradient_descent(W, Y, theta, alpha, iterations)

    printData(T, color)
    plt.plot(T, W.dot(theta), color = color , label = 'Linear Regression')

    return loss_history


# In[926]:


def gradientDes(loss_history, color):
    iterations = 1500;
    plt.plot(range(1, iterations + 1), loss_history, color = color)
    plt.rcParams["figure.figsize"] = (10,6)
    plt.grid()
    plt.xlabel('Number of Iterations')
    plt.ylabel('Cost (J)')
    plt.title('Convergence of Gradient Descent')


# In[927]:


printData(X, 'green')
printData(K, 'gold')
printData(Z, 'black')


# In[928]:


loss_history = graphRegression(X_0, X, 'green')


# In[929]:


gradientDes(loss_history, 'green')


# In[930]:


loss_history = graphRegression(X_1, K, 'gold')


# In[931]:


gradientDes(loss_history, 'gold')


# In[932]:


loss_history = graphRegression(X_2, Z, 'black')


# In[933]:


gradientDes(loss_history, 'black')


# In[934]:



#Problem 1: Q3

#X_1 or K had the steepest slope and therefore shows it required the least
#amount of iteratoins before reaching the correct value

#Problem 1: Q4

#The greater the learning rate, the steeper the curve. This allowed for less
#iterations. Furthermore, having too high of a learning rate, for example 
#2000, breaks the graph entirely due to dimensional errors. 


# In[935]:


def fourDTheta():
    X0 = np.ones((m,1))
    X1 = X.reshape(m,1)
    X2 = K.reshape(m,1)
    X3 = Z.reshape(m,1)
    
    X_3 = np.hstack((X0, X1, X2, X3))
    
    theta = np.zeros(4)
    iterations = 1500;
    alpha = 0.1;
    
    loss = compute_loss(X_3, Y, theta)
    theta, loss_history = gradient_descent(X_3, Y, theta, alpha, iterations)
    
    return theta, loss_history


# In[936]:


theta, loss_history = fourDTheta()
gradientDes(loss_history, 'red')


# In[937]:


# Poblem 2: Q3

# The learning rate must higher for a 2D graph becuase if the learning 
# rate is too low, then it won't stabalize before 1500 iterations.

# Problem 2: Q4

# Predict the value of y for new (X1, X2, X3) values (1, 1, 1), 
# for (2, 0, 4), and for (3, 2, 1)


# In[938]:


y_pred = theta[0] + (1)*theta[1] + (1)*theta[2] + (1)*theta[3]
y_pred


# In[939]:


y_pred = theta[0] + (2)*theta[1] + (0)*theta[2] + (4)*theta[3]
y_pred


# In[940]:


y_pred = theta[0] + (3)*theta[1] + (2)*theta[2] + (1)*theta[3]
y_pred

