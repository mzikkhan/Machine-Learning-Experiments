#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# In[22]:


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length','sepal-width','petal-length','petal-width','class']
dataset = pandas.read_csv(url, names = names)


# In[23]:


print(dataset.shape)


# In[24]:


print(dataset.head(30))


# In[25]:


print(dataset.describe())


# In[26]:


print(dataset.groupby('class').size())


# In[30]:


dataset.plot(kind = 'box', subplots = True, layout = (2,2), sharex = True, sharey = True)
plt.show()


# In[31]:


dataset.hist()
plt.show()


# In[32]:


scatter_matrix(dataset)
plt.show()


# In[34]:


array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 6
X_train,X_test, Y_train,Y_test = model_selection.train_test_split(X,Y, test_size = validation_size, random_state = seed)


# In[35]:


seed = 6
scoring = 'accuracy'


# In[41]:


models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits = 10)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv = kfold, scoring = scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[9]:





# In[10]:





# In[11]:




