# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 21:24:53 2022

@author: ashwi
"""
import pandas as pd
df = pd.read_csv("50_Startups.csv")
df.shape
df.dtypes
df.head()

df["State"].value_counts()

# Blanks
df.isnull().sum()
# finding duplicate rows
df.duplicated()
df[df.duplicated()] # hence no duplicates between the rows


# finding duplicate columns
df.columns.duplicated() # hence no duplicates between the column

#========================================================
# histogram 
df["R&D Spend"].hist()
df["R&D Spend"].skew()
# R&D spend is +ve skewness of (0.1240)

df["R&D Spend"].hist()
df["Administration"].skew()
# Administration is -ve skewness of (-0.4890)

df["Marketing Spend"].hist()
df["Marketing Spend"].skew()
# Marketing Spend have -ve skewness of (-0.0464)

df["Profit"].hist()
df["Profit"].skew()
# Profit is +ve skewness of (0.0232)

#==========================================================
# scatter plot
import matplotlib.pyplot as plt
plt.scatter(df["R&D Spend"],df["R&D Spend"],color="Black")
# +ve relationship this variable

plt.scatter(df["R&D Spend"],df["Marketing Spend"],color="Black")
# +ve relationship b\w this variable 

plt.scatter(df["R&D Spend"],df["Profit"],color="Black")
# +ve relationship b\w this variable

#============================================================
# boxplot
df.boxplot(column="R&D Spend",vert=False)

import numpy as np
Q1 = np.percentile(df["R&D Spend"],25)
Q2 = np.percentile(df["R&D Spend"],50)
Q3 = np.percentile(df["R&D Spend"],75)
IQR = Q3 - Q1
LW = Q1 - (1.5*IQR)
UW = Q3 + (1.5*IQR)
df[(df["R&D Spend"]<LW) | (df["R&D Spend"]>UW)]
len(df[(df["R&D Spend"]<LW) | (df["R&D Spend"]>UW)])
# in these variable there is zero outlaires


df.boxplot(column="Administration",vert=False)
import numpy as np
Q1 = np.percentile(df["Administration"],25)
Q2 = np.percentile(df["Administration"],50)
Q3 = np.percentile(df["Administration"],75)
IQR = Q3 - Q1
LW = Q1 - (1.5*IQR)
UW = Q3 + (1.5*IQR)
df[(df["Administration"]<LW) | (df["Administration"]>UW)]
len(df[(df["Administration"]<LW) | (df["Administration"]>UW)])
# in these variable there is zero outlaires

df.boxplot(column="Marketing Spend",vert=False)
import numpy as np
Q1 = np.percentile(df["Marketing Spend"],25)
Q2 = np.percentile(df["Marketing Spend"],50)
Q3 = np.percentile(df["Marketing Spend"],75)
IQR = Q3 - Q1
LW = Q1 - (1.5*IQR)
UW = Q3 + (1.5*IQR)
df[(df["Marketing Spend"]<LW) | (df["Marketing Spend"]>UW)]
len(df[(df["Marketing Spend"]<LW) | (df["Marketing Spend"]>UW)])
# in these variable there is zero outlaires

df.boxplot(column="Profit",vert=False)
import numpy as np
Q1 = np.percentile(df["Profit"],25)
Q2 = np.percentile(df["Profit"],50)
Q3 = np.percentile(df["Profit"],75)
IQR = Q3 - Q1
LW = Q1 - (1.5*IQR)
UW = Q3 + (1.5*IQR)
df[(df["Profit"]<LW) | (df["Profit"]>UW)]
len(df[(df["Profit"]<LW) | (df["Profit"]>UW)])
# in these variable there is one outlaires

# merging the out layer in lower wisker
import numpy as np
df["Profit"]=np.where(df["Profit"]>UW,UW,np.where(df["Profit"]<LW,LW,df["Profit"]))
df.boxplot(column="Profit",vert=False)
# Therefore one outlaire is merged in the lower wisker

#======================================================================
# Lable encoding
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df["State"] = LE.fit_transform(df["State"])
# lable encoding is done to the state 
df.head()
df.corr()
# therefore different types of  model are created based on co-relation 
#=====================================================================
# Split the variable as X and Y
Y = df["Profit"]
# X = df[["R&D Spend"]]   # Model-1
# X = df[["Marketing Spend"]] # Model-2
# X = df[["R&D Spend","Administration"]] # Model-3
X = df[["Marketing Spend","Administration"]] # Model-4
#=======================================================================
# Data partition
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.30,random_state=32)
X_train.shape,X_test.shape,Y_train.shape,Y_test.shape

#===================================================================
# model fitting 
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X_train,Y_train)

# B0
LR.intercept_

#B1
LR.coef_

# predictions
Y_pred_train = LR.predict(X_train)
Y_pred_train
Y_pred_test = LR.predict(X_test)
Y_pred_test
#===================================================================
# Matrics
from sklearn.metrics import mean_squared_error,r2_score
Training_error = mean_squared_error(Y_train, Y_pred_train)
testing_error = mean_squared_error(Y_test, Y_pred_test)

print("Training error",Training_error.round(3))
print("Testing error",testing_error.round(3))

import numpy as np
print("Root mean squared error", np.sqrt(Training_error).round(3))
print("Root mean squared error", np.sqrt(testing_error).round(3))

r2_train = r2_score(Y_train, Y_pred_train)
print("R square :", r2_train.round(3))


r2_test = r2_score(Y_test, Y_pred_test)
print("R square :", r2_test.round(3))
#=====================================================================
# validation set approch 
TrE = []
TsE = []
for i in range(1,201):
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.30,random_state=i)
    LR.fit(X_train,Y_train)
    Y_pred_train = LR.predict(X_train)
    Y_pred_test = LR.predict(X_test)
    TrE.append(mean_squared_error(Y_train,Y_pred_train))
    TsE.append(mean_squared_error(Y_test,Y_pred_test))

print(TrE)
print(TsE)

import numpy as np
np.mean(TrE)
np.mean(TsE)
#=====================================================================
# K-fold cross-validation
from sklearn.model_selection import KFold,cross_val_score
kfold = KFold(n_splits=10)
LR = LinearRegression()
train_result = abs(cross_val_score(LR, X_train, Y_train, cv=kfold,scoring="neg_mean_squared_error"))
test_result = abs(cross_val_score(LR, X_test, Y_test, cv=kfold,scoring="neg_mean_squared_error"))
train_result
test_result

train_result.mean()
test_result.mean()
#=====================================================================
# LOOCV
from sklearn.model_selection import LeaveOneOut,cross_val_score
loocv = LeaveOneOut()
LR =LinearRegression()
train_result1 = abs(cross_val_score(LR,X_train,Y_train,cv=loocv,scoring="neg_mean_squared_error"))
test_result1 = abs(cross_val_score(LR,X_test,Y_test,cv=loocv,scoring="neg_mean_squared_error"))
train_result1
test_result1

train_result1.mean()
test_result1.mean()
# therefore by see all the model validation techniques i did't notice any drastic reduced in error's so i decided that not to go with any validation techniques
#=====================================================================
import statsmodels.api as sma
Y_new = sma.add_constant(X)
Lm2 = sma.OLS(Y, Y_new).fit()
Lm2.summary()
# Therefore from above 4 models, model-3 is giving best results but P-value is getting more then 0.05 
# To check model is best or not VIF is doing
####################################  VIF  ####################################
import pandas as pd
df = pd.read_csv("50_Startups.csv")
df.head()
# loading the data
Y = df["Marketing Spend"]
X = df[["Administration"]]

# import linear regression
from sklearn.linear_model import LinearRegression
lm = LinearRegression().fit(X,Y)

# predict with our model coeffcients
Y_pred = lm.predict(X)

# monual calculations
import pandas as pd
import numpy as np
RSS = np.sum((Y_pred - Y)**2) # Residual sum of squares
Y_mean = np.mean(Y)
TSS = np.sum((Y - Y_mean)**2)
R2 = 1-(RSS/TSS)
print("R2 :", R2)

VIF = 1/(1-R2)
print("VIF value :",VIF)
# Therefore for model-3 VIF value got (1.06 ) and for model-4 VIF value we got (1.001)
# So I have salected the model-4 is my X 
#########################################################################################
################################### final model #########################################

Y = df["Profit"]
X = df[["Marketing Spend","Administration"]] # Model-4

# Data partition
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.30,random_state=32)
X_train.shape,X_test.shape,Y_train.shape,Y_test.shape


# model fitting 
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X_train,Y_train)

# B0
LR.intercept_

#B1
LR.coef_

# predictions
Y_pred_train = LR.predict(X_train)
Y_pred_train
Y_pred_test = LR.predict(X_test)
Y_pred_test
#===================================================================
# Matrics
from sklearn.metrics import mean_squared_error,r2_score
Training_error = mean_squared_error(Y_train, Y_pred_train)
testing_error = mean_squared_error(Y_test, Y_pred_test)

print("Training error",Training_error.round(3))
print("Testing error",testing_error.round(3))

import numpy as np
print("Root mean squared error", np.sqrt(Training_error).round(3))
print("Root mean squared error", np.sqrt(testing_error).round(3))

r2_train = r2_score(Y_train, Y_pred_train)
print("R square :", r2_train.round(3))


r2_test = r2_score(Y_test, Y_pred_test)
print("R square :", r2_test.round(3))






