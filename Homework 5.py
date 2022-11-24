#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# John Smith
# 800776897
# https://github.com/jsmit659/IntroMachineLearning/blob/main/
# SmithJohn_800776897_HW5.py


# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn import datasets

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


t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u)


# In[3]:


def model(t_u, w, b):
    return w * t_u + b


# In[4]:


def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c)**2
    return squared_diffs.mean()


# In[5]:


w = torch.ones(())
b = torch.zeros(()) 
t_p = model(t_u, w, b)
t_p


# In[6]:


loss = loss_fn(t_p, t_c)
loss


# In[7]:


delta = 0.1
loss_rate_of_change_w = (loss_fn(model(t_u, w + delta, b), t_c) -
loss_fn(model(t_u, w - delta, b), t_c)) / (2.0 * delta)


# In[8]:


learning_rate = 1e-2
w = w - learning_rate * loss_rate_of_change_w


# In[9]:


loss_rate_of_change_b = (loss_fn(model(t_u, w, b + delta), t_c) -
loss_fn(model(t_u, w, b - delta), t_c)) / (2.0 * delta)
b = b - learning_rate * loss_rate_of_change_b


# In[10]:


def model(t_u, w,b):
    return w*t_u+b


# In[11]:


def loss_fn(t_p,t_c):
    squared_diffs = (t_p-t_c)**2
    return squared_diffs.mean()


# In[12]:


def dloss_fn(t_p, t_c_):
    dsq_diffs = 2 * (t_p - t_c) /t_p.size(0)
    return dsq_diffs


# In[13]:


def dmodel_dw(t_u, w, b):
    return t_u


# In[14]:


def dmodel_db(t_u, w,b):
    return 1.0


# In[15]:


def grad_fn(t_u, t_c, t_p, w, b):
    dloss_dtp = dloss_fn(t_p, t_c)
    dloss_dw = dloss_dtp * dmodel_dw(t_u, w, b)
    dloss_db = dloss_dtp * dmodel_db(t_u, w, b)
    return torch.stack([dloss_dw.sum(), dloss_db.sum()])


# In[16]:


def training_loop(n_epochs, learning_rate, params, t_u, t_c):
    for epoch in range (1, n_epochs + 1):
        w, b = params
        t_p = model(t_u, w, b)
        loss = loss_fn(t_p, t_c)
        grad = grad_fn(t_u, t_c, t_p, w, b)
        params = params - learning_rate * grad
        print('Epoch %d, Loss %f' % (epoch, float(loss)))
    return params


# In[17]:


training_loop(
n_epochs = 100,
learning_rate = 1e-2,
params = torch.tensor([1.0,0.0]),
t_u = t_u,
t_c = t_c)


# In[18]:


training_loop(
n_epochs = 100,
learning_rate = 1e-4,
params = torch.tensor([1.0,0.0]),
t_u = t_u,
t_c = t_c)


# In[19]:


t_un = 0.1 *t_u


# In[20]:


training_loop(
n_epochs = 100,
learning_rate = 1e-2,
params = torch.tensor([1.0,0.0]),
t_u = t_un,
t_c = t_c)


# In[21]:


params = training_loop(
n_epochs = 5000,
learning_rate = 1e-2,
params = torch.tensor([1.0,0.0]),
t_u = t_un,
t_c = t_c,)
print_params = False
params


# In[22]:


from matplotlib import pyplot as plt
t_p = model(t_un, *params)
fig = plt.figure(dpi=600)
plt.xlabel=("Temperature (Fahrenheigh)")
plt.ylabel=("Temperature (Celsuis)")
plt.plot(t_u.numpy(), t_p.detach().numpy())
plt.plot(t_u.numpy(), t_c.numpy(),'o')


# In[23]:


def model(t_u, w,b):
    return w*t_u+b


# In[24]:


def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c)**2
    return squared_diffs.mean()


# In[25]:


params = torch.tensor([1.0,0.0], requires_grad=True)


# In[26]:


params = torch.tensor([1.0, 0.0], requires_grad=True)
loss = loss_fn(model(t_u, *params), t_c)
loss.backward()
params.grad


# In[27]:


def training_loop(n_epochs, learning_rate, params, t_u, t_c):
    for epoch in range(1, n_epochs+1):
        if params.grad is not None:
            params.grad.zero_()
            
        t_p = model(t_u, *params)
        loss = loss_fn(t_p,t_c)
        loss.backward()
        
        with torch.no_grad():
            params -= learning_rate * params.grad
            
        if epoch % 500 == 0:
            print('Epoch %d, Loss %f' % (epoch, float(loss)))
    return params


# In[28]:


training_loop(
n_epochs = 5000,
learning_rate = 1e-2,
params = torch.tensor([1.0,0.0], requires_grad=True),
t_u = t_un,
t_c = t_c)


# In[29]:


import torch.optim as optim
dir(optim)


# In[30]:


t_p = model(t_u, *params)
loss = loss_fn(t_p, t_c)
loss.backward()
params


# In[31]:


params = torch.tensor([1.0,0.0], requires_grad = True)
learning_rate = 1e-2
optimizer = optim.SGD([params], lr=learning_rate)
t_p = model(t_un, *params)
loss = loss_fn(t_p, t_c)
optimizer.zero_grad()
loss.backward()
optimizer.step()

params


# In[32]:


def training_loop(n_epochs, optimizer, params, t_u, t_c):
    for epoch in range(1, n_epochs+1):
        t_p = model(t_u, *params)
        loss=loss_fn(t_p,t_c)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            print('Epoch %d, Loss %f' % (epoch, float(loss)))
    return params


# In[33]:


params = torch.tensor([1.0,0.0], requires_grad=True)
learning_rate = 1e-2
optimizer = optim.SGD([params],lr=learning_rate)
training_loop(
n_epochs = 5000,
optimizer = optimizer,
params = params,
t_u = t_un,
t_c = t_c)


# # Number 1

# In[34]:


def new_model(t_u, w1, w2, b):
    return w2 * t_u **2 + w1 *t_u + b


# In[35]:


def training_loop(n_epochs, optimizer, params, t_u, t_c):
    for epoch in range(1, n_epochs+1):
        t_p = new_model(t_u, *params)
        loss=loss_fn(t_p,t_c);
        
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

    return params;


# In[36]:


params = torch.tensor([2.0,1.0,0.0])
params.requires_grad=True
learning_rate = 0.1
optimizer = optim.SGD([params],lr=learning_rate)

training_loop(
n_epochs = 5000,
optimizer = optimizer,
params = params,
t_u = t_un,
t_c = t_c)

from matplotlib import pyplot as plt
t_p = new_model(t_un, *params)
fig = plt.figure(dpi=100)
plt.xlabel=("Temperature (Fahrenheigh)")
plt.ylabel=("Temperature (Celsuis)")
plt.plot(t_u.numpy(), t_p.detach().numpy())
plt.plot(t_u.numpy(), t_c.numpy(),'o', color='blue')


# In[37]:


params = torch.tensor([2.0,1.0,0.0])
params.requires_grad=True
learning_rate = 0.1
optimizer = optim.SGD([params],lr=learning_rate)

training_loop(
n_epochs = 5000,
optimizer = optimizer,
params = params,
t_u = t_un,
t_c = t_c)
from matplotlib import pyplot as plt
t_p = new_model(t_un, *params)
fig = plt.figure(dpi=100)
plt.xlabel=("Temperature (Fahrenheigh)")
plt.ylabel=("Temperature (Celsuis)")
plt.plot(t_u.numpy(), t_p.detach().numpy())
plt.plot(t_u.numpy(), t_c.numpy(),'o', color='blue')


# In[38]:


params = torch.tensor([2.0,1.0,0.0])
params.requires_grad=True
learning_rate = 0.01
optimizer = optim.SGD([params],lr=learning_rate)

training_loop(
n_epochs = 5000,
optimizer = optimizer,
params = params,
t_u = t_un,
t_c = t_c)
from matplotlib import pyplot as plt
t_p = new_model(t_un, *params)
fig = plt.figure(dpi=100)
plt.xlabel=("Temperature (Fahrenheigh)")
plt.ylabel=("Temperature (Celsuis)")
plt.plot(t_u.numpy(), t_p.detach().numpy())
plt.plot(t_u.numpy(), t_c.numpy(),'o', color='blue')


# In[39]:


params = torch.tensor([2.0,1.0,0.0])
params.requires_grad=True
learning_rate = 0.001
optimizer = optim.SGD([params],lr=learning_rate)

training_loop(
n_epochs = 5000,
optimizer = optimizer,
params = params,
t_u = t_un,
t_c = t_c)
from matplotlib import pyplot as plt
t_p = new_model(t_un, *params)
fig = plt.figure(dpi=100)
plt.xlabel=("Temperature (Fahrenheigh)")
plt.ylabel=("Temperature (Celsuis)")
plt.plot(t_u.numpy(), t_p.detach().numpy())
plt.plot(t_u.numpy(), t_c.numpy(),'o', color='blue')


# In[40]:


params = torch.tensor([2.0,1.0,0.0])
params.requires_grad=True
learning_rate = 0.0001
optimizer = optim.SGD([params],lr=learning_rate)

training_loop(
n_epochs = 5000,
optimizer = optimizer,
params = params,
t_u = t_un,
t_c = t_c)
from matplotlib import pyplot as plt
t_p = new_model(t_un, *params)
fig = plt.figure(dpi=100)
plt.xlabel=("Temperature (Fahrenheigh)")
plt.ylabel=("Temperature (Celsuis)")
plt.plot(t_u.numpy(), t_p.detach().numpy())
plt.plot(t_u.numpy(), t_c.numpy(),'o', color='blue')


# 1.c Pick the best non-linear model and compare your final best loss against the linear model that we did during the lecture. For this, visualize the non-linear model against the linear model over the input dataset, as we did during the lecture. Is the actual result better or worse than our baseline linear model?
# 
# This model, using a learning rate of 0.0001, gives a better aproximation to the actual data as compared to the linear model. 

# Problem 2:
#     
#     

# In[41]:


import torch.nn as nn

device = torch.device("cuda:0")

housing = pd.DataFrame(pd.read_csv('Housing.csv'))

varlist = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']
def binary_map(x):
    return x.map({'yes' : 1, 'no' : 0, 'furnished' : 1, 'semi-furnished' : 0.5, 'unfurnished' : 0 })
housing[varlist] = housing[varlist].apply(binary_map)


# In[42]:


from sklearn.preprocessing import StandardScaler

features = ['price','area','bedrooms','bathrooms', 'stories', 'parking']
data = housing[features]
data = StandardScaler().fit_transform(data)
data


# In[43]:


raw_y = data[:, 0]
raw_x=data[:,1:6]
x=torch.from_numpy(raw_x)
y=torch.from_numpy(raw_y)


# In[44]:


n_samples = x.shape[0]
n_val = int(0.2 * n_samples)
shuffled_indices = torch.randperm(n_samples)
train_indices = shuffled_indices[:-n_val]
val_indices = shuffled_indices[-n_val:]


# n_samples = x.shape[0]
# n_val = int(0.2 * n_samples)
# indices_shuffled = torch.randperm(n_samples)
# indices_train = indices_shuffled[:-n_val]
# indices_val = indices_shuffled[-n_val:]

# In[45]:


def housing_model(X,W1,W2,W3,W4,W5,B):
    U=W5*X[:,4] + W4*X[:,3] + W3*X[:,2] + W2*X[:,1] + W1*X[:,0] + B
    return U

def loss_fn(Yp, Y):
    squared_diffs = (Yp - Y)**2
    return squared_diffs.mean()


# In[46]:


train_t_u = x[train_indices]
train_t_c = y[train_indices]
val_t_u = x[train_indices]
val_t_c = y[train_indices]
train_t_un = 0.1 * train_t_u
val_t_un = 0.1*val_t_u


# In[47]:


def training_loop(n_epochs, optimizer, params, train_t_u, val_t_u, 
                  train_t_c, val_t_c):
    for epoch in range(1,n_epochs+1):
        train_t_p = housing_model(train_t_u, *params)
        train_loss = loss_fn(train_t_p, train_t_c)
        
        val_t_p = housing_model(val_t_u, *params)
        val_loss = loss_fn(val_t_p, val_t_c)
        
        with torch.no_grad():
            val_t_p = housing_model(val_t_u, *params)
            val_loss = loss_fn(val_t_p, val_t_c)
            assert val_loss.requires_grad == False
        
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        if epoch <= 3 or epoch % 500 == 0:
            print (f"epoch {epoch}, Training loss {train_loss.item():.4f}," f" Validation loss {val_loss.item():.4f}")
                   
    return params


# In[48]:


def training_SGD(lr):
    params=torch.tensor([1.0,1.0,1.0,1.0,1.0,0.0],requires_grad=True)
    learning_rate=lr
    optimizer=optim.SGD([params],lr=learning_rate)

    training_loop(
        n_epochs = 5000,
        optimizer = optimizer,
        params = params,
        train_t_u = train_t_un,
        val_t_u = val_t_un,
        train_t_c = train_t_c,
        val_t_c = val_t_c)


# In[49]:


training_SGD(0.1)


# In[50]:


training_SGD(0.01)


# In[51]:


training_SGD(0.001)


# In[52]:


training_SGD(0.0001)


# In[ ]:





# In[53]:


def training_Adam(lr):
    params=torch.tensor([1.0,1.0,1.0,1.0,1.0,0.0],requires_grad=True)
    learning_rate=lr
    optimizer=optim.Adam([params],lr=learning_rate)

    training_loop(
        n_epochs = 5000,
        optimizer = optimizer,
        params = params,
        train_t_u = train_t_un,
        val_t_u = val_t_un,
        train_t_c = train_t_c,
        val_t_c = val_t_c)


# In[54]:


training_Adam(0.1)


# Number 3

# In[157]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import torch
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import torch.nn as nn

housing = pd.DataFrame(pd.read_csv('Housing.csv'))
features = ['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'parking']
housing = housing[features]
housing = StandardScaler().fit_transform(housing) 


# In[158]:


Y = housing[:, 0]
X = housing[:, 1:6]


# In[159]:


t_Y = torch.tensor(Y)
t_X = torch.tensor(X)
t_X.shape


# In[160]:


n_samples = t_X.shape[0]
n_val = int(0.2 * n_samples)

shuffled_indices = torch.randperm(n_samples)

train_indices = shuffled_indices[:-n_val]
val_indices = shuffled_indices[-n_val:]

train_indices, val_indices


# In[161]:


t_X_train = t_X[train_indices]
t_Y_train = t_Y[train_indices]

t_X_val = t_X[val_indices]
t_Y_val = t_Y[val_indices]

t_Xn_train = 0.1 * t_X_train
t_Xn_val = 0.1 * t_X_val


# In[162]:


linear_model = nn.Linear(1 , 1)

optimizer = optim.SGD(
    linear_model.parameters(), # <2>
    lr=0.001)

seq_model = nn.Sequential(
            nn.Linear(5, 8), # <1>
            nn.Tanh(),
            nn.Linear(8, 1)) # <2>

seq_model = seq_model.double()


# In[163]:


[param.shape for param in seq_model.parameters()]

for name, param in seq_model.named_parameters():
    print(name, param.shape)


# In[164]:


linear_model.weight

linear_model.bias
# In[165]:


x = torch.ones(1)


# In[166]:


linear_model(x)


# In[167]:


x = torch.ones(10, 1)
linear_model(x)


# In[168]:


linear_model = nn.Linear(1, 1)
optimizer = optim.SGD(
    linear_model.parameters(), 
    lr=1e-2)


# In[169]:


linear_model.parameters()
list(linear_model.parameters())


# In[170]:


def training_loop(n_epochs, optimizer, model, loss_fn, t_X_train, t_X_val,
                  t_Y_train, t_Y_val):
    for epoch in range(1, n_epochs + 1):
        t_p_train = model(t_X_train) # <1>
        loss_train = loss_fn(t_p_train, t_Y_train)

        t_p_val = model(t_X_val) # <1>
        loss_val = loss_fn(t_p_val, t_Y_val)
        
        optimizer.zero_grad()
        loss_train.backward() # <2>
        optimizer.step()

        if epoch == 1 or epoch % 50 == 0:
            print(f"Epoch {epoch}, Training loss {loss_train.item():.4f},"
                  f" Validation loss {loss_val.item():.4f}")


# In[171]:


import warnings
def loss_fn(Yp, Y):
    squared_diffs = (Yp - Y)**2
    return squared_diffs.mean()

optimizer = optim.SGD(seq_model.parameters(), lr=1e-3) # <1>

training_loop(
    n_epochs = 200, 
    optimizer = optimizer,
    model = seq_model,
    loss_fn = nn.MSELoss(),
    t_X_train = t_Xn_train,
    t_X_val = t_Xn_val, 
    t_Y_train = t_Y_train,
    t_Y_val = t_Y_val)

warnings
print()
print(linear_model.weight)
print(linear_model.bias)


# In[172]:


import torch.nn as nn
seq_model = nn.Sequential(
            nn.Linear(5, 8), # <1>
            nn.Tanh(),
            nn.Linear(8, 4), # <2>
            nn.Tanh(),
            nn.Linear(4, 2), # <3>
            nn.Tanh(),
            nn.Linear(2, 1)) 


# In[173]:


[param.shape for param in seq_model.parameters()]


# In[174]:


for name, param in seq_model.named_parameters():
    print(name, param.shape)


# In[175]:


from collections import OrderedDict

seq_model = nn.Sequential(OrderedDict([
    ('hidden_linear', nn.Linear(1, 10)),
    ('hidden_activation', nn.Tanh()),
    ('output_linear', nn.Linear(10, 1))
]))

seq_model


# In[176]:


for name, param in seq_model.named_parameters():
    print(name, param.shape)


# In[177]:


seq_model.output_linear.bias


# In[178]:


from collections import OrderedDict

seq_model = nn.Sequential(OrderedDict([
    ('hidden_linear', nn.Linear(1, 10)),
    ('hidden_activation', nn.Tanh()),
    ('output_linear', nn.Linear(10, 1))
]))

seq_model


# In[179]:


for name, param in seq_model.named_parameters():
    print(name, param.shape)


# In[180]:


seq_model.output_linear.bias


# In[ ]:





# In[ ]:





# In[ ]:




