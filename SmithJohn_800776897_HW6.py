#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# John Smith
# HW6
# 800776897
# 


# In[3]:


from torchvision import transforms
transforms.Normalize((0.4915, 0.4823, 0.4468),(0.2470, 0.2435, 0.2616))
from torchvision import datasets
data_path = '\My Drive\Fall 2022\Machine Learning\HW6'
cifar10 = datasets.CIFAR10(data_path, train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4915, 0.4823, 0.4468),(0.2470, 0.2435, 0.2616))]))
cifar10_val = datasets.CIFAR10(data_path, train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4915, 0.4823, 0.4468),(0.2470, 0.2435, 0.2616))]))


# In[6]:


import torch.nn as nn
CIFAR_model = nn.Sequential(
            nn.Linear(3072,512), # <1>
            nn.Tanh(),
            nn.Linear(512, 10), # <2>
            nn.Tanh(),
            nn.LogSoftmax(dim=1)) 


# In[8]:


import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device


# In[9]:


CIFAR_model.to(device)
loss_fn = nn.NLLLoss()


# In[10]:


import torch.optim as optim
optimizer = optim.SGD(CIFAR_model.parameters(),lr=1e-3)


# In[11]:


train_loader = torch.utils.data.DataLoader(cifar10, batch_size=64, shuffle=True)
num_epochs = 300


# In[12]:


for epoch in range(num_epochs):
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        batch_size = imgs.shape[0]
        outputs = CIFAR_model(imgs.view(batch_size, -1))
        loss = loss_fn(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print("Epoch: %d, Loss: %f" % (epoch, float(loss)))


# In[23]:


CIFAR_model_2 = nn.Sequential(
            nn.Linear(3072,512), # <1>
            nn.Tanh(),
            nn.Linear(512, 256), # <2>
            nn.Tanh(),
            nn.Linear(256,128), # <1>
            nn.Tanh(),
            nn.Linear(128, 10), # <2>
            nn.Tanh(),
            nn.LogSoftmax(dim=1)) 
CIFAR_model_2.to(device)
loss_fn = nn.NLLLoss()


# In[24]:


optimizer = optim.SGD(CIFAR_model_2.parameters(),lr=1e-3)


# In[25]:


train_loader = torch.utils.data.DataLoader(cifar10, batch_size=64, shuffle=True)
num_epochs = 300


# In[26]:


for epoch in range(num_epochs):
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        batch_size = imgs.shape[0]
        outputs = CIFAR_model_2(imgs.view(batch_size, -1))
        loss = loss_fn(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print("Epoch: %d, Loss: %f" % (epoch, float(loss)))


# In[ ]:


# From the above data, we can tell that the data is slightly overfitting 


# In[30]:


model = nn.Sequential(
            nn.Conv2d(3,16,kernel_size = 3, padding=1), 
            nn.Tanh(),
            nn.MaxPool2d(2),
            nn.Conv2d(16,8,kernel_size=3, padding=1),
            nn.Tanh(), 
            nn.MaxPool2d(2),
            nn.Linear(8*8*8, 32), 
            nn.Tanh(),
            nn.Linear(32,10)
)


# In[40]:


import datetime
def training_loop(n_epochs, optimizer, model, loss_fn, train_loader):
  for epoch in range(1, n_epochs + 1):
    loss_train = 0.0
    for imgs, labels in train_loader:
      imgs = imgs.to(device=device)
      labels = labels.to(device=device)
      outputs = model(imgs)
      loss = loss_fn(outputs, labels)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      loss_train += loss.item()
    print('{} Epoch {}, Training loss {}'.format(
    datetime.datetime.now(), epoch,
    loss_train / len(train_loader)))
    


# In[43]:


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.act1 = nn.Tanh()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.act2 = nn.Tanh()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(8 * 8 * 8, 32)
        self.act3 = nn.Tanh()
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        out = self.pool1(self.act1(self.conv1(x)))
        out = self.pool2(self.act2(self.conv2(out)))
        out = out.view(-1, 8 * 8 * 8) # <1>
        out = self.act3(self.fc1(out))
        out = self.fc2(out)
        return out


# In[50]:


train_loader = torch.utils.data.DataLoader(cifar10, batch_size=64,
                                        shuffle=True)
model = Net().to(device=device)  #  <2>
optimizer = optim.SGD(model.parameters(), lr=1e-2)  #  <3>
loss_fn = nn.CrossEntropyLoss()  #  <4>

training_loop(  # <5>
    n_epochs = 300,
    optimizer = optimizer,
    model = model,
    loss_fn = loss_fn,
    train_loader = train_loader,
)


# In[51]:


train_loader = torch.utils.data.DataLoader(cifar10, batch_size=64,
                                          shuffle=False)
val_loader = torch.utils.data.DataLoader(cifar10_val, batch_size=64,
                                        shuffle=False)
validate(model, train_loader, val_loader)


# In[45]:


def validate(model, train_loader, val_loader):
  for name, loader in [("train", train_loader), ("val", val_loader)]:
    correct = 0
    total = 0
    with torch.no_grad():
      for imgs, labels in loader:
          imgs, labels = imgs.to(device), labels.to(device)
          batchsize = imgs.shape[0]
          outputs = model(imgs)
          _, predicted = torch.max(outputs, dim=1)
          total += labels.shape[0]
          correct += int((predicted == labels).sum())
    print("Accuracy {}: {:.2f}".format(name , correct / total))


# In[46]:


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.act1 = nn.Tanh()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.act2 = nn.Tanh()
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(8, 3, kernel_size=3, padding=1)
        self.act3 = nn.Tanh()
        self.pool3 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(8 * 8 * 8, 32)
        self.act3 = nn.Tanh()
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        out = self.pool1(self.act1(self.conv1(x)))
        out = self.pool2(self.act2(self.conv2(out)))
        out = out.view(-1, 8 * 8 * 8) # <1>
        out = self.act3(self.fc1(out))
        out = self.fc2(out)
        return out


# In[47]:


train_loader = torch.utils.data.DataLoader(cifar10, batch_size=64,
                                        shuffle=True)
model = Net().to(device=device)  #  <2>
optimizer = optim.SGD(model.parameters(), lr=1e-2)  #  <3>
loss_fn = nn.CrossEntropyLoss()  #  <4>

training_loop(  # <5>
    n_epochs = 300,
    optimizer = optimizer,
    model = model,
    loss_fn = loss_fn,
    train_loader = train_loader,
)


# In[49]:


train_loader = torch.utils.data.DataLoader(cifar10, batch_size=64,
                                          shuffle=False)
val_loader = torch.utils.data.DataLoader(cifar10_val, batch_size=64,
                                        shuffle=False)
validate(model, train_loader, val_loader)


# In[ ]:


# This dataset shows signs of little to no overfitting

