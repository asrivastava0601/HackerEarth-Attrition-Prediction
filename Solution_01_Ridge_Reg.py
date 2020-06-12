#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 20:06:35 2020

@author: abhinavsrivastava
"""



import numpy as np
import pandas as pd
import math

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt


train_data = pd.read_csv("Train.csv")


train_data.isnull().values.any()

train_data.isnull().sum()

desc = train_data.describe()

stats = train_data.iloc[:,2:23].apply(pd.Series.value_counts)


age = train_data['Age'].describe()


import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.figure(figsize= (10,8))
sns.distplot(train_data['Age'].dropna(), hist = True ,kde=False, color='darkred' ,bins=20)
mean= train_data['Age'].mean()
median= train_data['Age'].median()
plt.axvline(mean, color='r', linestyle='--')
plt.axvline(median, color='g', linestyle='-')
plt.legend({'Mean':mean,'Median':median})
plt.title("Age Distribution")

plt.figure(figsize= (8,8))
sns.boxplot(x= 'Gender', y= 'Age', data= train_data)

plt.figure(figsize= (8,8))
sns.boxplot(x= 'Relationship_Status', y= 'Age', data= train_data)

mean_age_on_g_and_rs = train_data.groupby(['Gender', 'Relationship_Status'])['Age'].mean().to_frame()

median_age_on_g_rs = train_data.groupby(['Gender', 'Relationship_Status'])['Age'].median().to_frame()

mean_age_on_rs = train_data.groupby(['Relationship_Status'])['Age'].mean().to_frame()

median_age_on_rs = train_data.groupby(['Relationship_Status'])['Age'].median().to_frame()

#~~~~~~~~~~ Fill missing Age column with median wrt the gender and relatioship status ~~~~~~~~~

def fill_missing_age (cols):
    Age = cols[0]
    Gender = cols[1]
    Relationship_Status= cols[2]
    
    if (pd.isnull(Age)):
        if (Gender == 'M'):
            return 37
        elif (Gender == 'F'):
            if (Relationship_Status == 'Married'):
                return 37
            else: return 38
    else: return Age

train_data['Age'].isnull().sum()

train_data['Age']= train_data[['Age','Gender','Relationship_Status']].apply(fill_missing_age, axis=1)

train_data['Age'].isnull().sum()


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

train_data['Time_of_service'].isnull().sum()

time_of_service = train_data['Time_of_service'].describe()

plt.figure(figsize= (8,8))
sns.distplot(train_data['Time_of_service'], hist= True, kde= False)
mean_ts = train_data['Time_of_service'].mean()
median_ts = train_data['Time_of_service'].median()
plt.axvline(mean_ts, color='r', linestyle='--')
plt.axvline(median_ts, color='g', linestyle='-')
plt.legend({'Mean':mean_ts,'Median':median_ts})
plt.title("Time_of_service Distribution")

from statsmodels.graphics.gofplots import qqplot

qqplot(train_data['Time_of_service'], line= 's')
plt.show()

#~~~~~~~~~~~~~~~~~~~~ Fill missing Time of service with median ~~~~~~~~~~~~~~~~
train_data['Time_of_service'].isnull().sum()

median_ts = train_data['Time_of_service'].median()

train_data['Time_of_service'] = train_data['Time_of_service'].fillna(median_ts)

train_data['Time_of_service'].isnull().sum()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~~~ Pay Scale ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

train_data['Pay_Scale'].describe()

train_data['Pay_Scale'].isnull().sum()

sns.distplot(train_data['Pay_Scale'].dropna(), hist = True, kde= False)

median_ps = math.floor(train_data['Pay_Scale'].median())


mode_ps = math.floor(train_data['Pay_Scale'].mode())


train_data['Pay_Scale'] = train_data['Pay_Scale'].fillna(mode_ps)

train_data['Pay_Scale'].isnull().sum()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~~~ Work life balance ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
train_data['Work_Life_balance'].isnull().sum()

train_data['Work_Life_balance'].describe()

sns.distplot(train_data['Work_Life_balance'].dropna(), hist = True, kde= False)

train_data['Work_Life_balance'].value_counts()

mode_wlb = math.floor(train_data['Work_Life_balance'].mode())

train_data['Work_Life_balance'] = train_data['Work_Life_balance'].fillna(mode_wlb)

train_data['Work_Life_balance'].isnull().sum()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~~~ VAR2 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

train_data['VAR2'].describe()

train_data['VAR2'].isnull().sum()

sns.distplot(train_data['VAR2'].dropna(), hist = True, kde= False)

var2 = train_data['VAR2'].dropna()

var2.isnull().sum()

mean_var2 = train_data['VAR2'].mean()

train_data['VAR2'].median()


train_data['VAR2'] = train_data['VAR2'].fillna(mean_var2)

train_data['VAR2'].isnull().sum()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#~~~~~~~~~~~~~~~~~~~~~~ VAR4 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

train_data['VAR4'].describe()

train_data['VAR4'].isnull().sum()

sns.distplot(train_data['VAR4'].dropna(), hist = True, kde= False)

var4 = train_data['VAR4'].dropna()

var4.unique()

var4.isnull().sum()

median_var4 = math.floor(train_data['VAR4'].median())

train_data['VAR4'] = train_data['VAR4'].fillna(median_var4)

train_data['VAR4'].isnull().sum()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


###############################################################################
#   All the missing values have been filled.
#       Now start of the feature selection.
###############################################################################



temp_data = train_data.drop(['Employee_ID', 'Attrition_rate','Hometown','Unit', 'Decision_skill_possess'], axis =1)

temp_y = train_data['Attrition_rate']


from sklearn.preprocessing import LabelEncoder

lb_enc = LabelEncoder()

temp_data.head()

temp_data['Gender'] =lb_enc.fit_transform(temp_data['Gender'])

temp_data.head()

temp_data['Relationship_Status'] =lb_enc.fit_transform(temp_data['Relationship_Status'])

temp_data.head()

temp_data['Compensation_and_Benefits'] =lb_enc.fit_transform(temp_data['Compensation_and_Benefits'])

temp_data.head()


#~~~~~~~~~~~~~~~~~~ Univariate analysis ~~~~~~~~~~~~~~~~~~~~

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

X = temp_data  #independent columns
y = temp_y

X_no_numerical = temp_data

X_no_numerical = X_no_numerical.drop(['Age','Time_of_service','growth_rate','VAR2','VAR3'], axis =1)

############# f_regression ##########

bestfeatures2 = SelectKBest(score_func= f_regression, k=10)
fit2 = bestfeatures2.fit(X_no_numerical,y)

dfscores2 = pd.DataFrame(fit2.scores_)
dfcolumns2 = pd.DataFrame(X_no_numerical.columns)

featureScores2 = pd.concat([dfcolumns2,dfscores2],axis=1)
featureScores2.columns = ['Specs','Score']  #naming the dataframe columns

featureScores2


X_numerical = temp_data[['Age','Time_of_service','growth_rate','VAR2','VAR3']]

bestfeatures4 = SelectKBest(score_func= f_regression, k=2)
fit4 = bestfeatures4.fit(X_numerical,y)

dfscores4 = pd.DataFrame(fit4.scores_)
dfcolumns4 = pd.DataFrame(X_numerical.columns)

featureScores4 = pd.concat([dfcolumns4,dfscores4],axis=1)
featureScores4.columns = ['Specs','Score']  #naming the dataframe columns

featureScores4


##### Correlation Matrix #######
t2 = pd.concat([X_no_numerical,temp_y], axis =1)

#get correlations of each features in dataset
corrmat = t2.corr()

top_corr_features = corrmat.index

plt.figure(figsize=(20,20))

#plot heat map
g=sns.heatmap(t2[top_corr_features].corr(),annot=True,cmap="RdYlGn")

########################## Label Encoding #####################################

Education_Level = pd.get_dummies(X['Education_Level'],drop_first=True, prefix = 'Education_Level')

Time_since_promotion = pd.get_dummies(X['Time_since_promotion'],drop_first=True, prefix = 'Time_since_promotion')

Travel_Rate = pd.get_dummies(X['Travel_Rate'],drop_first=True, prefix = 'Travel_Rate')

Post_Level = pd.get_dummies(X['Post_Level'],drop_first=True, prefix = 'Post_Level')

Pay_Scale = pd.get_dummies(X['Pay_Scale'],drop_first=True, prefix = 'Pay_Scale')

Compensation_and_Benefits = pd.get_dummies(X['Compensation_and_Benefits'],drop_first=True, prefix = 'Compensation_and_Benefits')

Work_Life_balance = pd.get_dummies(X['Work_Life_balance'],drop_first=True, prefix = 'Work_Life_balance')

VAR1 = pd.get_dummies(X['VAR1'],drop_first=True, prefix = 'VAR1')

VAR4 = pd.get_dummies(X['VAR4'],drop_first=True, prefix = 'VAR4')

VAR5 = pd.get_dummies(X['VAR5'],drop_first=True, prefix = 'VAR5')

VAR6 = pd.get_dummies(X['VAR6'],drop_first=True, prefix = 'VAR6')

VAR7 = pd.get_dummies(X['VAR4'],drop_first=True, prefix = 'VAR7')


X = X.drop(['Education_Level','Time_since_promotion','Travel_Rate','Post_Level','Pay_Scale','Compensation_and_Benefits','Work_Life_balance','VAR1','VAR4','VAR5','VAR6','VAR7'], axis=1)


X = pd.concat([X,Education_Level,Time_since_promotion,Travel_Rate,Post_Level,Pay_Scale,Compensation_and_Benefits,Work_Life_balance,VAR1,VAR4,VAR5,VAR6,VAR7], axis=1)



########## Model with selected feature based on f_regressor values ############

new_X_data = temp_data[['Gender','Compensation_and_Benefits','Work_Life_balance','VAR2']]


new_X_data = pd.concat([new_X_data,Compensation_and_Benefits,Work_Life_balance], axis=1)

new_X_data = new_X_data.drop(['Compensation_and_Benefits','Work_Life_balance'], axis=1)

new_X_data = pd.concat([new_X_data,temp_data['Time_of_service']], axis=1)



########################### Train Test split ##################################

new_X_train, new_X_test, new_Y_train, new_Y_test = train_test_split(new_X_data, y, test_size=0.2, random_state= 42)

new_regressor = LinearRegression()

new_regressor.fit(new_X_train,new_Y_train)

new_y_pred = new_regressor.predict(new_X_test)


new_rmse = sqrt(mean_squared_error(new_Y_test, new_y_pred))

print("Root mean squared error (based on f_regressor values model): ", new_rmse)

new_result = 100* max(0, 1- new_rmse)

print("\n Result for all feature model (based on f_regressor values model): ", new_result)


new_y_pred_train = new_regressor.predict(new_X_train)

new_rmse_train = sqrt(mean_squared_error(new_Y_train, new_y_pred_train))
print("\nTraining - Root mean squared error (based on f_regressor values model): ", new_rmse_train)

new_result_train = 100* max(0, 1- new_rmse_train)

print("\nTraining - Result for all feature model (based on f_regressor values model): ", new_result_train)


#~~~~~~~~~~~~~~~~~ Cross validation of above model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from sklearn.model_selection import cross_val_score

lin_regressor= LinearRegression()

mse=cross_val_score(lin_regressor,new_X_data,y,scoring='neg_mean_squared_error',cv= 5)

mean_mse=np.mean(mse)

print(mean_mse)


#~~~~~~~~~~~~~~~~~~~~~~~~~~  Ridge and Lasso ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

ridge=Ridge()


parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}

ridge_regressor= GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=5)

ridge_regressor.fit(new_X_data,y)


print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)



from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
lasso=Lasso()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
lasso_regressor=GridSearchCV(lasso,parameters,scoring='neg_mean_squared_error',cv=5)

lasso_regressor.fit(new_X_data,y)


print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)



prediction_lasso=lasso_regressor.predict(new_X_test)
prediction_ridge=ridge_regressor.predict(new_X_test)

mse_ridge = sqrt(mean_squared_error(new_Y_test, prediction_ridge))

print("\n Root mean squared error ridge: ", mse_ridge)

result_ridge = 100* max(0, 1- mse_ridge)

print("\n Result for ridge: ", result_ridge)



mse_lasso = sqrt(mean_squared_error(new_Y_test, prediction_lasso))

print("\n Root mean squared error lasso: ", mse_lasso)

result_lasso = 100* max(0, 1- mse_lasso)

print("\n Result for lasso: ", result_lasso)



################# Prediction on the test set for submission ###################

test_data = pd.read_csv("Test.csv")

predict_testset = test_data[['Gender','Compensation_and_Benefits','Work_Life_balance','VAR2']]


predict_testset.isnull().sum()

predict_testset.dtypes


predict_testset['Work_Life_balance'].value_counts()

mode_test_wlb = math.floor(predict_testset['Work_Life_balance'].mode())

predict_testset['Work_Life_balance'].fillna(mode_test_wlb, inplace= True)


predict_testset.isnull().sum()


mean_test_var2 = predict_testset['VAR2'].mean()

predict_testset['VAR2'].fillna(mean_test_var2, inplace= True)


predict_testset.isnull().sum()


predict_testset['Gender'] =lb_enc.fit_transform(predict_testset['Gender'])


predict_testset['Compensation_and_Benefits'] =lb_enc.fit_transform(predict_testset['Compensation_and_Benefits'])


median_test_ts = test_data['Time_of_service'].median()

test_data['Time_of_service'] = test_data['Time_of_service'].fillna(median_test_ts)


test_Compensation_and_Benefits = pd.get_dummies(predict_testset['Compensation_and_Benefits'],drop_first=True, prefix = 'Compensation_and_Benefits')

test_Work_Life_balance = pd.get_dummies(predict_testset['Work_Life_balance'],drop_first=True, prefix = 'Work_Life_balance')


predict_testset = predict_testset.drop(['Compensation_and_Benefits','Work_Life_balance'], axis =1)

predict_testset = pd.concat([predict_testset, test_Compensation_and_Benefits, test_Work_Life_balance, test_data['Time_of_service']], axis =1)


##### ~~~~~~~~~~~~~~ Prediction using Ridge regressor ~~~~~~~~~~~~~~~~~~~~~~~~~

predict_testset_ridge = pd.DataFrame(ridge_regressor.predict(predict_testset))

predict_testset_ridge = predict_testset_ridge.rename(columns ={0: 'Attrition_rate'})

submission04 = pd.concat([test_data['Employee_ID'], predict_testset_ridge], axis=1)

submission04.to_csv("Solution04.csv", index = False)

