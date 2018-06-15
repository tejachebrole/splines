
# coding: utf-8

# ###  Wage Data Analysis
# In this application (which we refer to as the Wage data set throughout this
# book), we examine a number of factors that relate to wages for a group of
# males from the Atlantic region of the United States. In particular, we wish
# to understand the association between an employee’s age and education, as
# well as the calendar year, on his wage. Consider, for example, the left-hand
# panel of Figure 1.1, which displays wage versus age for each of the individuals
# in the data set. There is evidence that wage increases with age but then
# decreases again after approximately age 60. The blue line, which provides
# an estimate of the average wage for a given age, makes this trend clearer.

# In[86]:

import pandas as pd
data=pd.read_csv('wages.csv')


# In[87]:

data.head()


# In[88]:

data.describe()


# In[89]:

data.dtypes[data.dtypes==object]


# In[90]:

pd.options.display.float_format = '{:.2f}'.format
pd.set_option('display.precision',3)


# In[91]:

data.groupby(by='race')['age','wage'].mean()


# In[92]:

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
from IPython.display import display
import seaborn as sns
#fig, axs = plt.subplots(1,2)
axs=sns.boxplot(x='maritl', y='wage', data=data,showfliers=False,linewidth=2) #ax =
plt.show()
axs2=sns.boxplot(x='education', y='wage', data=data,showfliers=False,linewidth=2)
plt.show()
#display(ax)


# Linear Regression

# In[93]:

# divide the data into test and train data
from sklearn.model_selection import train_test_split
data_y=data['wage']
data_x=data[['age','race','education','maritl','health','jobclass']]
train_x, valid_x, train_y, valid_y = train_test_split(data_x, data_y, test_size=0.33, random_state = 1)


# In[94]:

# handle the categorical variables through patsy
# Fitting linear regression model
from sklearn.linear_model import LinearRegression
from patsy import dmatrices, dmatrix
c=dmatrix("train_x['age']+train_x['race']+train_x['education']+train_x['jobclass']+            train_x['maritl']+train_x['health']-1",train_x)
model = LinearRegression()
model.fit(c,train_y)
print(model.coef_)
print(model.intercept_)


# In[95]:

# Prediction on validation dataset
d=dmatrix("valid_x['age']+valid_x['race']+valid_x['education']+valid_x['jobclass']            + valid_x['maritl']+valid_x['health']-1",valid_x)
pred = model.predict(d)
from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(valid_y, pred))
print(rms)


# In[96]:

# 3D Visualisation
# We will use 70 plots between minimum and maximum values of valid_x for plotting

import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
sns.boxplot(x="race", y="wage", data=data)


# Polynomial Regression

# In[117]:

from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
poly = PolynomialFeatures(degree=2)
c_ = poly.fit_transform(c)
clf = linear_model.LinearRegression()
clf.fit(c_, train_y)
#print(clf.coef_);


# In[118]:

d_=poly.fit_transform(d)
pred = clf.predict(d_)
from sklearn.metrics import mean_squared_error
from math import sqrt
rms = sqrt(mean_squared_error(valid_y, pred))
print(rms)


# Using Splines

# In[ ]:



