import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn import  linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
import statsmodels.api as sm
from matplotlib.pyplot import plot 
pd.set_option('display.max_columns', None) #This option displays all the columns 


#########################
##House price prediction
#########################
kc_house_data = pd.read_csv(r'D:\Google Drive\Training\Book\0.Chapters\Chapter5 Feature Selection\Datasets\kc_house_data\kc_house_data.csv')

#Get an idea on number of rows and columns
print(kc_house_data.shape)

#Print the column names
print(kc_house_data.columns)

#Print the column types
print(kc_house_data.dtypes)

#Additional Details
kc_house_data.info()

#Summary
all_cols_summary=kc_house_data.describe()
print(round(all_cols_summary,2))

#Model Building

#import statsmodels.formula.api as sm
#model1 = sm.ols(formula='price ~ bedrooms+bathrooms+sqft_living+sqft_lot+floors+waterfront+view+condition+grade+sqft_above+sqft_basement+yr_built+yr_renovated+zipcode+lat+long+sqft_living15+sqft_lot15', data=kc_house_data)
#fitted1 = model1.fit()
#fitted1.summary()

#Defining X data
X = kc_house_data[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15']]

y = kc_house_data['price']

from sklearn  import model_selection
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y ,test_size=0.2, random_state=55)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

import sklearn 
model_1 = sklearn.linear_model.LinearRegression()
model_1.fit(X_train, y_train)

#Coefficients and Intercept
print(model_1.intercept_)
print(model_1.coef_)

#Rsquared Calculation on Train data
from sklearn import metrics
y_pred_train=model_1.predict(X_train)
print(metrics.r2_score(y_train,y_pred_train))

#Rsquared Calculation on test data
y_pred_test=model_1.predict(X_test)
print(metrics.r2_score(y_test,y_pred_test))

#RSquared
print("R-Squared on Train data : ", metrics.r2_score(y_train,y_pred_train))
print("R-Squared on Test data : ", metrics.r2_score(y_test,y_pred_test))

#MAD
print("MAD on Train data : ", round(np.mean(np.abs(y_train - y_pred_train)),2))
print("MAD on Test data : ", round(np.mean(np.abs(y_test - y_pred_test)),2))

#MAPE
print("MAPE on Train data : ", round(np.mean(np.abs(y_train - y_pred_train)/y_train),2))
print("MAPE on Test data : ", round(np.mean(np.abs(y_test - y_pred_test)/y_test),2))

#RMSE
print("RMSE on Train data : ", round(math.sqrt(np.mean(np.abs(y_train - y_pred_train)**2)),2))
print("RMSE on Test data : ", round(math.sqrt(np.mean(np.abs(y_test - y_pred_test)**2)),2))

#describe price
round(kc_house_data.price.describe())

#########################
#Credit Risk Data
##########################
import pandas as pd
credit_risk_data = pd.read_csv(r'D:\Google Drive\Training\Book\0.Chapters\Chapter5 Feature Selection\Datasets\loans_data\credit_risk_data_v1.csv')

#Get an idea on number of rows and columns
print(credit_risk_data.shape)

#Print the column names
print(credit_risk_data.columns)

#Print the column types
print(credit_risk_data.dtypes)

#Additional Details
credit_risk_data.info()

#Summary
pd.set_option('display.max_columns', None) #This option displays all the columns 

all_cols_summary=credit_risk_data.describe()
print(round(all_cols_summary,2))


#Defining X data
X = credit_risk_data[['Credit_Limit', 'Late_Payments_Count',
       'Card_Utilization_Percent', 'Age', 'Debt_to_income_ratio',
       'Monthly_Income', 'Num_loans_personal_loans', 'Family_dependents']]

y = credit_risk_data['Bad']

from sklearn  import model_selection
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y ,test_size=0.2, random_state=55)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

#Building the model
from sklearn.linear_model import LogisticRegression
model_2= LogisticRegression(solver="lbfgs")
model_2.fit(X_train,y_train)

#Coefficients and Intercept
print(model_2.intercept_)
print(model_2.coef_)

##Confusion Matrix Calculation on Train data
from sklearn.metrics import confusion_matrix

y_pred_train=model_2.predict(X_train)
cm1 = confusion_matrix(y_train,y_pred_train)
print(cm1)

##Accuracy on Train data
accuracy1=(cm1[0,0]+cm1[1,1])/(cm1[0,0]+cm1[0,1]+cm1[1,0]+cm1[1,1])
print(accuracy1)

##Confusion matrix on test data
y_pred_test=model_2.predict(X_test)
cm2 = confusion_matrix(y_test,y_pred_test)
print(cm2)

#####Accuracy on Test data
accuracy2=(cm2[0,0]+cm2[1,1])/(cm2[0,0]+cm2[0,1]+cm2[1,0]+cm2[1,1])
print(accuracy2)

#Frequency of target variable
credit_risk_data['Bad'].value_counts()

##Sensitivity on train data
Sensitivity1=cm1[0,0]/(cm1[0,0]+cm1[0,1])
print(round(Sensitivity1,3))

##Specificity on train data
Specificity1=cm1[1,1]/(cm1[1,0]+cm1[1,1])
print(round(Specificity1,3))


##Sensitivity on test data
Sensitivity2=cm2[0,0]/(cm2[0,0]+cm2[0,1])
print(round(Sensitivity2,3))

##Specificity on test data
Specificity2=cm2[1,1]/(cm2[1,0]+cm2[1,1])
print(round(Specificity2,3))

#####Probability predictions
y_pred_prob=model_2.predict_proba(X_train)
print(y_pred_prob.shape)
print(y_pred_prob)
print(y_pred_prob[0,])
print(y_pred_prob[0,0])
print(y_pred_prob[0,1])
print(y_pred_prob[0:5,1])
print(y_pred_prob[:,1])

y_pred_prob_1=y_pred_prob[:,1]

## Default Threshold 0.5
threshold=0.5
y_pred_class=y_pred_prob_1*0
y_pred_class[y_pred_prob_1>threshold]=1
print(y_pred_class)

##Confusion Matrix and accuracy
cm3 = confusion_matrix(y_train,y_pred_class)
print("confusion Matrix with Threshold ",  threshold,  "\n",cm3)
accuracy3=(cm3[0,0]+cm3[1,1])/(cm3[0,0]+cm3[0,1]+cm3[1,0]+cm3[1,1])
print("Accuracy is ", round(accuracy3,3))

##Sensitivity and Specificity on Train data
Sensitivity3=cm3[0,0]/(cm3[0,0]+cm3[0,1])
print("Sensitivity is", round(Sensitivity3,3))

Specificity3=cm3[1,1]/(cm3[1,0]+cm3[1,1])
print("Specificity is ", round(Specificity3,3))


## New Threshold 0.2
threshold=0.2
y_pred_class=y_pred_prob_1*0
y_pred_class[y_pred_prob_1>threshold]=1

##Confusion Matrix and accuracy
cm3 = confusion_matrix(y_train,y_pred_class)
print("confusion Matrix with Threshold ",  threshold,  "\n",cm3)
accuracy3=(cm3[0,0]+cm3[1,1])/(cm3[0,0]+cm3[0,1]+cm3[1,0]+cm3[1,1])
print("Accuracy is ", round(accuracy3,3))

##Sensitivity and Specificity on Train data
Sensitivity3=cm3[0,0]/(cm3[0,0]+cm3[0,1])
print("Sensitivity is", round(Sensitivity3,3))

Specificity3=cm3[1,1]/(cm3[1,0]+cm3[1,1])
print("Specificity is ", round(Specificity3,3))



## New Threshold 0.1
threshold=0.1
y_pred_class=y_pred_prob_1*0
y_pred_class[y_pred_prob_1>threshold]=1

##Confusion Matrix and accuracy
cm3 = confusion_matrix(y_train,y_pred_class)
print("confusion Matrix with Threshold ",  threshold,  "\n",cm3)
accuracy3=(cm3[0,0]+cm3[1,1])/(cm3[0,0]+cm3[0,1]+cm3[1,0]+cm3[1,1])
print("Accuracy is ", round(accuracy3,3))

##Sensitivity and Specificity on Train data
Sensitivity3=cm3[0,0]/(cm3[0,0]+cm3[0,1])
print("Sensitivity is", round(Sensitivity3,3))

Specificity3=cm3[1,1]/(cm3[1,0]+cm3[1,1])
print("Specificity is ", round(Specificity3,3))

####################
#####ROC AUC
####################

## New Threshold 0.1
threshold=0.01
y_pred_class=y_pred_prob_1*0
y_pred_class[y_pred_prob_1>threshold]=1

##Confusion Matrix and accuracy
cm3 = confusion_matrix(y_train,y_pred_class)
print("confusion Matrix with Threshold ",  threshold,  "\n",cm3)
accuracy3=(cm3[0,0]+cm3[1,1])/(cm3[0,0]+cm3[0,1]+cm3[1,0]+cm3[1,1])
print("Accuracy is ", round(accuracy3,3))

##Sensitivity and Specificity on Train data
Sensitivity3=cm3[0,0]/(cm3[0,0]+cm3[0,1])
print("Sensitivity is", round(Sensitivity3,3))

Specificity3=cm3[1,1]/(cm3[1,0]+cm3[1,1])
print("Specificity is ", round(Specificity3,3))

##ROC Curve Creation
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, y_pred_prob_1)
plt.figure(figsize=(10,10))
plt.title('ROC Curve',fontsize=15)
plt.plot(false_positive_rate, true_positive_rate)
plt.plot([0,1],[0,1],'r--')
plt.ylabel('True Positive Rate(Sensitivity)',fontsize=15)
plt.xlabel('False Positive Rate(1-Specificity)',fontsize=15)
plt.show()


###Area under Curve-AUC
auc = auc(false_positive_rate, true_positive_rate)
print(auc)

##F1 Score
from sklearn.metrics import f1_score

## Threshold 0.5
threshold=0.5
y_pred_class=y_pred_prob_1*0
y_pred_class[y_pred_prob_1>threshold]=1
print("threshold=0.5 f1_score ",f1_score(y_train, y_pred_class))

## Threshold 0.2
threshold=0.2
y_pred_class=y_pred_prob_1*0
y_pred_class[y_pred_prob_1>threshold]=1
print("threshold=0.2 f1_score ",f1_score(y_train, y_pred_class))

########################
####Cross-Validation
########################

###Test data cross-validation

diabetes_data= pd.read_csv(r'D:\Google Drive\Training\Book\0.Chapters\Chapter5 Feature Selection\Datasets\pima\diabetes.csv')

#Get an idea on number of rows and columns
print(diabetes_data.shape)

#Print the column names
print(diabetes_data.columns)

#Print the column types
print(diabetes_data.dtypes)

#Additional Details
diabetes_data.info()

#Summary
pd.set_option('display.max_columns', None) #This option displays all the columns 

all_cols_summary=diabetes_data.describe()
print(round(all_cols_summary,2))


#Defining X data
X = diabetes_data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness','Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]

y = diabetes_data[['Outcome']]

#Train and Test data creation
from sklearn  import model_selection
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y ,test_size=0.2, random_state=33)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

## Model Building 
from sklearn.tree import DecisionTreeClassifier
diabetes_tree1= DecisionTreeClassifier()
diabetes_tree1.fit(X_train, y_train)

#Calculate Accuracy on Train and Test data
print("Max Depth = None")
print("Train data Accuracy", diabetes_tree1.score(X_train, y_train))
print("Test data Accuracy", diabetes_tree1.score(X_test, y_test))

## Model Building with pruning parameters
from sklearn.tree import DecisionTreeClassifier
diabetes_tree1= DecisionTreeClassifier(max_depth=1)
diabetes_tree1.fit(X_train, y_train)

#Calculate Accuracy on Train and Test data
print("Max Depth = 1")
print("Train data Accuracy", diabetes_tree1.score(X_train, y_train))
print("Test data Accuracy", diabetes_tree1.score(X_test, y_test))

## Model Building with pruning parameters
print("Max Depth = 6")
from sklearn.tree import DecisionTreeClassifier
diabetes_tree1= DecisionTreeClassifier(max_depth=5)
diabetes_tree1.fit(X_train, y_train)

#Calculate Accuracy on Train and Test data
print("Train data Accuracy", diabetes_tree1.score(X_train, y_train))
print("Test data Accuracy", diabetes_tree1.score(X_test, y_test))

## Model Building with pruning parameters
print("Max Depth = 3")
from sklearn.tree import DecisionTreeClassifier
diabetes_tree1= DecisionTreeClassifier(max_depth=3)
diabetes_tree1.fit(X_train, y_train)

#Calculate Accuracy on Train and Test data
print("Train data Accuracy", diabetes_tree1.score(X_train, y_train))
print("Test data Accuracy", diabetes_tree1.score(X_test, y_test))


## Model Building with pruning parameters
print("Max Depth = 2")
from sklearn.tree import DecisionTreeClassifier
diabetes_tree1= DecisionTreeClassifier(max_depth=2)
diabetes_tree1.fit(X_train, y_train)

#Calculate Accuracy on Train and Test data
print("Train data Accuracy", diabetes_tree1.score(X_train, y_train))
print("Test data Accuracy", diabetes_tree1.score(X_test, y_test))

####K-fold cross-validation
diabetes_tree_KF = DecisionTreeClassifier(max_depth=3)
#Simple K-Fold cross validation. 10 folds.
from sklearn.model_selection import KFold
kfold = KFold(n_splits=10)

## Checking the accuracy of model on 10-folds
from sklearn import model_selection
acc10 = model_selection.cross_val_score(diabetes_tree_KF,X, y,cv=kfold)
print(acc10)
print(acc10.mean())

#### Train – Validation – Holdout cross-validation
from sklearn  import model_selection

## Split overall data into Train and Test split
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y ,test_size=0.3, random_state=99)

## Split Test data into Validation and Holdout data split
X_val, X_hold, y_val, y_hold = model_selection.train_test_split(X_test, y_test ,test_size=0.5 , random_state=11)

print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)
print(X_hold.shape)
print(y_hold.shape)


## Model Building with pruning parameters
print("Max Depth 6")
from sklearn.tree import DecisionTreeClassifier
diabetes_tree1= DecisionTreeClassifier(max_depth=6)
diabetes_tree1.fit(X_train, y_train)

#Calculate Accuracy on Train and Test data
print("Train data Accuracy", diabetes_tree1.score(X_train, y_train))
print("Validation data Accuracy", diabetes_tree1.score(X_val, y_val))


## Model Building with pruning parameters
print("Max Depth 1")
from sklearn.tree import DecisionTreeClassifier
diabetes_tree1= DecisionTreeClassifier(max_depth=1)
diabetes_tree1.fit(X_train, y_train)

#Calculate Accuracy on Train and Test data
print("Train data Accuracy", diabetes_tree1.score(X_train, y_train))
print("Validation data Accuracy", diabetes_tree1.score(X_val, y_val))


## Model Building with pruning parameters
print("Max Depth 3")
from sklearn.tree import DecisionTreeClassifier
diabetes_tree1= DecisionTreeClassifier(max_depth=3)
diabetes_tree1.fit(X_train, y_train)

#Calculate Accuracy on Train and Test data
print("Train data Accuracy", diabetes_tree1.score(X_train, y_train))
print("Validation data Accuracy", diabetes_tree1.score(X_val, y_val))

#Final Model and Result
print("Max Depth 3")
print("Train data Accuracy", diabetes_tree1.score(X_train, y_train))
print("Validation data Accuracy", diabetes_tree1.score(X_val, y_val))
print("Holdout data Accuracy", diabetes_tree1.score(X_hold, y_hold))

###Grid Search

from sklearn.model_selection import GridSearchCV
grid_param={'max_depth': range(1,10,1), 'max_leaf_nodes': range(2,30,1)}
clf_tree=DecisionTreeClassifier()
clf=GridSearchCV(clf_tree,grid_param)
clf.fit(X_train,y_train)

# examine the best model

# Single best score achieved across all params (min_samples_split)
print(clf.best_score_)

# Dictionary containing the parameters (min_samples_split) used to generate that score
print(clf.best_params_)

# Actual model object fit with those best parameters
# Shows default parameters that we did not specify
print(clf.best_estimator_)

grid_result_tree= clf.best_estimator_
print("Train data Accuracy", grid_result_tree.score(X_train, y_train))
print("Validation data Accuracy", grid_result_tree.score(X_val, y_val))


##########################
##Feature Engineering Tips and Tricks
############################

#Defining X data
X = kc_house_data[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15']]

y = kc_house_data['price']

from sklearn  import model_selection
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y ,test_size=0.2, random_state=55)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

import sklearn 
model_1 = sklearn.linear_model.LinearRegression()
model_1.fit(X_train, y_train)

#Coefficients and Intercept
print(model_1.intercept_)
print(model_1.coef_)

#Rsquared Calculation on Train data
from sklearn import metrics
y_pred_train=model_1.predict(X_train)
print(metrics.r2_score(y_train,y_pred_train))

#Rsquared Calculation on test data
y_pred_test=model_1.predict(X_test)
print(metrics.r2_score(y_test,y_pred_test))

#RMSE
print("RMSE on Train data : ", round(math.sqrt(np.mean(np.abs(y_train - y_pred_train)**2)),2))
print("RMSE on Test data : ", round(math.sqrt(np.mean(np.abs(y_test - y_pred_test)**2)),2))


###The Dummy Variable 

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,10))
sns.boxplot( x=kc_house_data["bedrooms"],y=kc_house_data["price"])
plt.title('Bedrooms vs House Price', fontsize=20)

plt.figure(figsize=(10,10))
sns.boxplot( x=kc_house_data["bathrooms"],y=kc_house_data["price"])
plt.title('Bathrooms vs House Price', fontsize=20)

plt.figure(figsize=(10,10))
sns.boxplot( x=kc_house_data["floors"],y=kc_house_data["price"])
plt.title('Floors vs House Price', fontsize=20)

plt.figure(figsize=(10,10))
sns.boxplot( x=kc_house_data["waterfront"],y=kc_house_data["price"])
plt.title('Waterfront vs House Price', fontsize=20)

plt.figure(figsize=(10,10))
sns.boxplot( x=kc_house_data["view"],y=kc_house_data["price"])
plt.title('View vs House Price', fontsize=20)

plt.figure(figsize=(10,10))
sns.boxplot( x=kc_house_data["condition"],y=kc_house_data["price"])
plt.title('Condition vs House Price', fontsize=20)

plt.figure(figsize=(10,10))
sns.boxplot( x=kc_house_data["grade"],y=kc_house_data["price"])
plt.title('Grade vs House Price', fontsize=20)

plt.figure(figsize=(10,10))
sns.boxplot( x=kc_house_data["zipcode"],y=kc_house_data["price"])
plt.title('Zipcode vs House Price', fontsize=20)

###The Dummy Variable Creation

print(kc_house_data.shape)
categorical_vars=['bedrooms', 'bathrooms', 'floors', 'waterfront', 'view', 'condition', 'grade', 'zipcode']

from sklearn.preprocessing import OneHotEncoder
encoding=OneHotEncoder()
encoding.fit(kc_house_data[categorical_vars])
onehotlabels = encoding.transform(kc_house_data[categorical_vars]).toarray()
onehotlabels_data=pd.DataFrame(onehotlabels)

print(kc_house_data.shape)

kc_house_data1 = kc_house_data.drop(categorical_vars,axis = 1)
print(kc_house_data1.shape)

kc_house_data_onehot=kc_house_data1.join(onehotlabels_data)
print(kc_house_data_onehot.shape)

###Model buiding with one hot encoded data

#Defining X data
col_names = kc_house_data_onehot.columns.values
print(col_names)

x_col_names=col_names[3:]
print(x_col_names)

X = kc_house_data_onehot[x_col_names]
y = kc_house_data_onehot['price']

from sklearn  import model_selection
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y ,test_size=0.2, random_state=55)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

import sklearn 
model_1 = sklearn.linear_model.LinearRegression()
model_1.fit(X_train, y_train)

#Coefficients and Intercept
print(model_1.intercept_)
print(model_1.coef_)

#Rsquared Calculation on Train data
from sklearn import metrics
y_pred_train=model_1.predict(X_train)
print("Train data R-Squared : ", metrics.r2_score(y_train,y_pred_train))

#Rsquared Calculation on test data
y_pred_test=model_1.predict(X_test)
print("Test data R-Squared : " , metrics.r2_score(y_test,y_pred_test))

#RMSE
print("RMSE on Train data : ", round(math.sqrt(np.mean(np.abs(y_train - y_pred_train)**2)),2))
print("RMSE on Test data : ", round(math.sqrt(np.mean(np.abs(y_test - y_pred_test)**2)),2))


### Handling Longitude and Latitude

###'House Price vs Longitude and Latitude'
bubble_col= kc_house_data["price"] > kc_house_data["price"].quantile(0.7)

import matplotlib.pyplot as plt
plt.figure(figsize=(12,12))
plt.scatter(kc_house_data["long"],kc_house_data["lat"], c=bubble_col,cmap="RdYlGn",s=10)
plt.title('House Price vs Longitude and Latitude', fontsize=20)
plt.xlabel('Longitude', fontsize=15)
plt.ylabel('Latitude', fontsize=15)
plt.show()

# Lets take the center of all high priced houses and low priced houses

high_long_mean=kc_house_data["long"][bubble_col].mean()
high_lat_mean=kc_house_data["lat"][bubble_col].mean()


import matplotlib.pyplot as plt
plt.figure(figsize=(12,12))
plt.scatter(kc_house_data["long"],kc_house_data["lat"], c=bubble_col,cmap="RdYlGn",s=10)
plt.scatter(high_long_mean,high_lat_mean, c="black", s=1000)
plt.title('House Price vs Longitude and Latitude', fontsize=20)
plt.xlabel('Longitude', fontsize=15)
plt.ylabel('Latitude', fontsize=15)
plt.show()

##Distance from center to every house
kc_house_data["High_cen_distance"]=np.sqrt((kc_house_data["long"] - high_long_mean) ** 2 + (kc_house_data["lat"] - high_lat_mean) ** 2)

plt.figure(figsize=(15,15))
plt.scatter(kc_house_data["High_cen_distance"],np.log(kc_house_data["price"]))
plt.title('House Price vs Distance from center', fontsize=20)
plt.xlabel('Distance from center', fontsize=15)
plt.ylabel('log(house price)', fontsize=15)

#Defining X data
col_names = kc_house_data.columns.values
print(col_names)

x_col_names=col_names[3:]
print(x_col_names)

X = kc_house_data[x_col_names]
y = kc_house_data['price']

from sklearn  import model_selection
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y ,test_size=0.2, random_state=55)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

import sklearn 
model_1 = sklearn.linear_model.LinearRegression()
model_1.fit(X_train, y_train)

#Coefficients and Intercept
print(model_1.intercept_)
print(model_1.coef_)

#Rsquared Calculation on Train data
from sklearn import metrics
y_pred_train=model_1.predict(X_train)
print("Train data R-Squared : ", metrics.r2_score(y_train,y_pred_train))

#Rsquared Calculation on test data
y_pred_test=model_1.predict(X_test)
print("Test data R-Squared : " , metrics.r2_score(y_test,y_pred_test))

#RMSE
print("RMSE on Train data : ", round(math.sqrt(np.mean(np.abs(y_train - y_pred_train)**2)),2))
print("RMSE on Test data : ", round(math.sqrt(np.mean(np.abs(y_test - y_pred_test)**2)),2))

###############################
##5.7.3 Handling date variables
###############################

#Print the column names
print(kc_house_data.columns)
date_vars = ['date', 'yr_built', 'yr_renovated']
kc_house_dates=kc_house_data[date_vars]
kc_house_dates.head()

kc_house_dates['sale_year'] = np.int64([d[0:4] for d in kc_house_dates["date"]])
kc_house_dates['sale_month'] = np.int64([d[4:6] for d in kc_house_dates["date"]])
kc_house_dates['day_sold'] = np.int64([d[6:8] for d in kc_house_dates["date"]])
kc_house_dates['age_of_house'] = kc_house_dates['sale_year'] - kc_house_dates['yr_built']
kc_house_dates['Ind_renovated'] = kc_house_dates['yr_renovated']>0

plt.figure(figsize=(10,10))
sns.boxplot( x=kc_house_dates['sale_year'],y=kc_house_data["price"])
plt.title('Sale_year vs House Price', fontsize=20)

plt.figure(figsize=(10,10))
sns.boxplot( x=kc_house_dates['sale_month'],y=kc_house_data["price"])
plt.title('Sale_month vs House Price', fontsize=20)

plt.figure(figsize=(10,10))
sns.boxplot( x=kc_house_dates['day_sold'],y=kc_house_data["price"])
plt.title('Day_sold vs House Price', fontsize=20)

plt.figure(figsize=(10,10))
plt.scatter(kc_house_dates["age_of_house"],kc_house_data["price"])
plt.title('Age_of_house vs House Price', fontsize=20)

plt.figure(figsize=(10,10))
sns.boxplot( x=kc_house_dates['Ind_renovated'],y=kc_house_data["price"])
plt.title('Ind_renovated vs House Price', fontsize=20)

##Model building with date variables 
kc_house_dates1=kc_house_dates.drop(date_vars, axis=1) #keep only newly derived variables
kc_house_with_dates=kc_house_data.join(kc_house_dates1)
print(kc_house_with_dates.shape)

###Model building with date variables
#Defining X data
col_names = kc_house_with_dates.columns.values
print(col_names)

x_col_names=col_names[3:]
print(x_col_names)

X = kc_house_with_dates[x_col_names]
y = kc_house_with_dates['price']

from sklearn  import model_selection
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y ,test_size=0.2, random_state=55)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

import sklearn 
model_1 = sklearn.linear_model.LinearRegression()
model_1.fit(X_train, y_train)

#Coefficients and Intercept
print(model_1.intercept_)
print(model_1.coef_)

#Rsquared Calculation on Train data
from sklearn import metrics
y_pred_train=model_1.predict(X_train)
print("Train data R-Squared : ", metrics.r2_score(y_train,y_pred_train))

#Rsquared Calculation on test data
y_pred_test=model_1.predict(X_test)
print("Test data R-Squared : " , metrics.r2_score(y_test,y_pred_test))

#RMSE
print("RMSE on Train data : ", round(math.sqrt(np.mean(np.abs(y_train - y_pred_train)**2)),2))
print("RMSE on Test data : ", round(math.sqrt(np.mean(np.abs(y_test - y_pred_test)**2)),2))


###########################
####Transformations

grid_plot1= sns.PairGrid(kc_house_data, y_vars=["price"], x_vars=["sqft_living", "sqft_lot"], height=5)
grid_plot1.map(sns.regplot)

grid_plot2 = sns.PairGrid(kc_house_data, y_vars=["price"], x_vars=["sqft_above", "sqft_basement"], height=5)
grid_plot2.map(sns.regplot)

grid_plot3 = sns.PairGrid(kc_house_data, y_vars=["price"], x_vars=["sqft_living15","sqft_lot15"], height=5)
grid_plot3.map(sns.regplot)

#Histogram on target variable
plt.figure(figsize=(10,10))
sns.distplot(kc_house_data["price"])
plt.title('House Price distribution', fontsize=20)

#Log tranformation
kc_house_data["log_price"]=np.log(kc_house_data["price"])
plt.figure(figsize=(10,10))
sns.distplot(kc_house_data["log_price"])
plt.title('log(House Price) distribution', fontsize=20)

###Model building after Transformations
#Defining X data
X = kc_house_data[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15']]

y = kc_house_data['log_price']

from sklearn  import model_selection
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y ,test_size=0.2, random_state=55)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

import sklearn 
model_1 = sklearn.linear_model.LinearRegression()
model_1.fit(X_train, y_train)

#Coefficients and Intercept
print(model_1.intercept_)
print(model_1.coef_)

#Rsquared Calculation on Train data
from sklearn import metrics
y_pred_train=model_1.predict(X_train)
print("Train data R-Squared : ", metrics.r2_score(y_train,y_pred_train))

#Rsquared Calculation on test data
y_pred_test=model_1.predict(X_test)
print("Test data R-Squared : " , metrics.r2_score(y_test,y_pred_test))

#RMSE
print("RMSE on Train data : ", round(math.sqrt(np.mean(np.abs(y_train - y_pred_train)**2)),2))
print("RMSE on Test data : ", round(math.sqrt(np.mean(np.abs(y_test - y_pred_test)**2)),2))

################################
#####Dealing with Class imbalance
################################

import pandas as pd
credit_risk_data = pd.read_csv(r'D:\Google Drive\Training\Book\0.Chapters\Chapter5 Feature Selection\Datasets\loans_data\credit_risk_data_v1.csv')

print("Actual Data :", credit_risk_data.shape)

#Frequency count on target column
print("Overall Data")
freq=credit_risk_data['Bad'].value_counts()
print(freq)
print((freq/freq.sum())*100)

#Classwise data
credit_risk_class0 = credit_risk_data[credit_risk_data['Bad'] == 0]
credit_risk_class1 = credit_risk_data[credit_risk_data['Bad'] == 1]

print("Class0 Actual :", credit_risk_class0.shape)
print("Class1 Actual  :", credit_risk_class1.shape)

##Undersampling of claa-0
## Consider half of class-0
credit_risk_class0_under = credit_risk_class0.sample(int(0.5*len(credit_risk_class0)))
print("Class0 Undersample :", credit_risk_class0_under.shape)

##Oversampling of Class-1 
# Lets increase the size by four times
credit_risk_class1_over = credit_risk_class1.sample(4*len(credit_risk_class1),replace=True)
print("Class1 Oversample :", credit_risk_class1_over.shape)

#Concatenate to create the final balanced data
credit_risk_balanced=pd.concat([credit_risk_class0_under,credit_risk_class1_over])
print("Final Balanced Data :", credit_risk_balanced.shape)

#Frequency count on target column in the balanced data
print("Balanced Data")
freq=credit_risk_balanced['Bad'].value_counts()
print(freq)
print((freq/freq.sum())*100)

#All the datasets and their shapes
print("Actual Data :", credit_risk_data.shape)
print("Class0 Actual :", credit_risk_class0.shape)
print("Class1 Actual  :", credit_risk_class1.shape)
print("Class0 Undersample :", credit_risk_class0_under.shape)
print("Class1 Oversample :", credit_risk_class1_over.shape)
print("Final Balannced Data :", credit_risk_balanced.shape)

#Model building on balanced data

#Defining X data
X = credit_risk_balanced[['Credit_Limit', 'Late_Payments_Count',
       'Card_Utilization_Percent', 'Age', 'Debt_to_income_ratio',
       'Monthly_Income', 'Num_loans_personal_loans', 'Family_dependents']]

y = credit_risk_balanced['Bad']

from sklearn  import model_selection
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y ,test_size=0.2, random_state=55)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

#Building the model
from sklearn.linear_model import LogisticRegression
model_2= LogisticRegression(solver='lbfgs')
model_2.fit(X_train,y_train)

#Coefficients and Intercept
print(model_2.intercept_)
print(model_2.coef_)

##Confusion Matrix Calculation on Train data

from sklearn.metrics import confusion_matrix

y_pred_train=model_2.predict(X_train)
cm1 = confusion_matrix(y_train,y_pred_train)
print("Confusion Matrix  on Train Data")
print(cm1)

##Accuracy on Train data
accuracy1=(cm1[0,0]+cm1[1,1])/(cm1[0,0]+cm1[0,1]+cm1[1,0]+cm1[1,1])
print("Accuracy on Train data ",accuracy1)

#Sensitivity and Specificity on Train data
Sensitivity1=cm1[0,0]/(cm1[0,0]+cm1[0,1])
print("Sensitivity Train data ", round(Sensitivity1,3))

Specificity1=cm1[1,1]/(cm1[1,0]+cm1[1,1])
print("Specificity Train data ",round(Specificity1,3))


##Confusion matrix on test data
y_pred_test=model_2.predict(X_test)
cm2 = confusion_matrix(y_test,y_pred_test)
print("Confusion Matrix  on Test Data")
print(cm2)

#####Accuracy on Test data
accuracy2=(cm2[0,0]+cm2[1,1])/(cm2[0,0]+cm2[0,1]+cm2[1,0]+cm2[1,1])
print("Accuracy on Test data ", accuracy2)

#Sensitivity and Specificity on Test data
Sensitivity2=cm2[0,0]/(cm2[0,0]+cm2[0,1])
print("Sensitivity Test data ",round(Sensitivity2,3))

#Specificity
Specificity2=cm2[1,1]/(cm2[1,0]+cm2[1,1])
print("Specificity Test data ", round(Specificity2,3))


