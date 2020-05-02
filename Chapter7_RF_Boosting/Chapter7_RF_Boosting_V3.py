import pandas as pd
import sklearn as sk
import numpy as np
import scipy as sp

from sklearn  import model_selection
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc, f1_score,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', None) #This option displays all the columns 

#Importing dataset
car_train=pd.read_csv(r"D:\Google Drive\Training\Book\0.Chapters\Chapter7 Random Forest and Boosting\Datasets\car_accidents\car_sensors.csv")

#Get an idea on number of rows and columns
print(car_train.shape)

#Print the column names
print(car_train.columns)

#Print the column types
print(car_train.info())

##Data Exploration
#Summary
all_cols_summary=car_train.describe()
print(round(all_cols_summary,2))

#Target variable
print(car_train['safe'].value_counts())

##Defining train and test data
features=car_train.columns.values[1:]
print(features)
X = car_train[features]
y = car_train['safe']

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y ,test_size=0.2, random_state=55)

print("X_train Shape ",X_train.shape)
print("y_train Shape ", y_train.shape)
print("X_test Shape ",X_test.shape)
print("y_test Shape ", y_test.shape)

###buildng Decision tree on the training data ####
D_tree = tree.DecisionTreeClassifier(max_depth=7)
D_tree.fit(X_train,y_train)

#####Accuracy on train data ####
tree_predict1=D_tree.predict(X_train)
cm1 = confusion_matrix(y_train,tree_predict1)
accuracy_train=(cm1[0,0]+cm1[1,1])/sum(sum(cm1))
print("Decison Tree Accuracy on Train data = ", round(accuracy_train,2) )

#####Accuracy on test data ####
tree_predict2=D_tree.predict(X_test)
cm2 = confusion_matrix(y_test,tree_predict2)
accuracy_test=(cm2[0,0]+cm2[1,1])/sum(sum(cm2))
print("Decison Tree Accuracy on Test data = ", round(accuracy_test,2) )

##AUC on Train data
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, tree_predict1)
auc_train = auc(false_positive_rate, true_positive_rate)
print("Decison Tree AUC on Train data = ", round(auc_train,2) )

##AUC on Test data
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, tree_predict2)
auc_test = auc(false_positive_rate, true_positive_rate)
print("Decison Tree AUC on Test data = ", round(auc_test,2) )

################
####Building Random Forest Model
R_forest=RandomForestClassifier(n_estimators=300, max_features=4, max_depth=10)
R_forest.fit(X_train,y_train)

#####Accuracy on train data ####
forest_predict1=R_forest.predict(X_train)
cm1 = confusion_matrix(y_train,forest_predict1)
accuracy_train=(cm1[0,0]+cm1[1,1])/sum(sum(cm1))
print("Random Forest Accuracy on Train data = ", round(accuracy_train,2) )

####Accuracy on test data ####
forest_predict2=R_forest.predict(X_test)
cm2 = confusion_matrix(y_test,forest_predict2)
accuracy_test=(cm2[0,0]+cm2[1,1])/sum(sum(cm2))
print("Random Forest Accuracy on Test data = ", round(accuracy_test,2) )

##AUC on Train data
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, forest_predict1)
auc_train = auc(false_positive_rate, true_positive_rate)
print("Random Forest AUC on Train data =  ", round(auc_train,2) )

##AUC on Test data
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, forest_predict2)
auc_test= auc(false_positive_rate, true_positive_rate)
print("Random Forest AUC on Test data =  ", round(auc_test,2) )

#################################

#####GBM illustration

import pandas as pd
pets_data = pd.read_csv(r"D:\Google Drive\Training\Book\0.Chapters\Chapter7 Random Forest and Boosting\Datasets\Pet_adoption\adoption.csv")
pets_data.columns.values
pets_data.head(10)


X=pets_data[["cust_age"]]
y=h=pets_data['adopted_pet']


for i in range (1,21):
    
    #Model and predictions 
    boost_model=GradientBoostingClassifier(n_estimators=i,learning_rate=1, max_depth=1)
    boost_model.fit(X,y)
    pets_data["itaration_result"]=boost_model.predict_proba(X)[:,1]
    boost_predict= boost_model.predict(X)
    
    #Graph
    fig = plt.figure()
    plt.rcParams["figure.figsize"] = (7,5)
    plt.title(['learning_rate=1', 'Iteration :', i ], fontsize=20)
    ax1 = fig.add_subplot(111)
    ax1.scatter(pets_data["cust_age"],pets_data["adopted_pet"], s=50, c='b', marker="x")
    ax1.scatter(pets_data["cust_age"],pets_data["itaration_result"], s=50, c='r', marker="o")
    ax1.set_xlabel('cust_age')
    ax1.set_ylabel('adopted_pet')
    
    #SSE and Accuracy
    print("SSE : ", sum((pets_data["itaration_result"] - y)**2))
    accuracy=f1_score(y, boost_predict, average='micro')
    print("Accuracy : ", accuracy)

## New learning rate learning_rate=0.1
for i in range (1,202):
    
    #Model and predictions 
    boost_model=GradientBoostingClassifier(n_estimators=i,learning_rate=0.1, max_depth=1)
    boost_model.fit(X,y)
    pets_data["itaration_result"]=boost_model.predict_proba(X)[:,1]
    boost_predict= boost_model.predict(X)
    
    #Graph
    if(np.mod(i, 10) ==1):
        fig = plt.figure()
        plt.rcParams["figure.figsize"] = (7,5)
        plt.title(['learning_rate=0.1', 'Iteration :', i ], fontsize=20)
        ax1 = fig.add_subplot(111)
        ax1.scatter(pets_data["cust_age"],pets_data["adopted_pet"], s=50, c='b', marker="x")
        ax1.scatter(pets_data["cust_age"],pets_data["itaration_result"], s=50, c='r', marker="o")
        ax1.set_xlabel('cust_age')
        ax1.set_ylabel('adopted_pet')
        
    #SSE and Accuracy
    print("SSE : ", sum((pets_data["itaration_result"] - y)**2))
    accuracy=f1_score(y, boost_predict, average='micro')
    print("Accuracy : ", accuracy)

#################################
#####Case Study- Income Prediction from Census Data 

income = pd.read_csv(r"D:\Google Drive\Training\Book\0.Chapters\Chapter7 Random Forest and Boosting\Datasets\Adult_Census_Income\Adult_Income.csv")

#Get an idea on number of rows and columns
print(income.shape)

#Print the column names
print(income.columns)

#Print the column types
print(income.info())

##Data Exploration
#Summary
all_cols_summary=income.describe()
print(round(all_cols_summary,2))

##Categorical Variables Exploration
categorical_vars=income.select_dtypes(include=['object']).columns
print(categorical_vars)

##Frequency tables for all the categorical columns
for col in categorical_vars:
    print("\n\nFrequency Table for the column ", col )
    print(income[col].value_counts())
    
##Data Cleaning and Feature Engineering
    
##workclass
income["workclass"] = income["workclass"].replace(['?','Never-worked','Without-pay'], 'Other')  
print(income["workclass"] .value_counts())
    
##marital.status
income["marital.status"] = income["marital.status"].replace(['Never-married','Divorced','Separated','Widowed'], 'Not-married')
print(income["marital.status"] .value_counts())
    
##occupation
income["occupation"] = income["occupation"].replace(['?'], 'Other-service')  
print(income["occupation"] .value_counts())
 
##Country
freq_country=income["native.country"].value_counts()
less_frequent= freq_country[freq_country <100].index
print(less_frequent)

income["native.country"]=income["native.country"].replace([less_frequent], 'Other')
income["native.country"] = income["native.country"].replace(['?'], 'Other')  
print(income["native.country"].value_counts())

## Converting Gender into class-0 and Class-1
print(income["sex"].value_counts())
income['sex']=income['sex'].map({'Male': 0, 'Female': 1})


## Converting target into class-0 and Class-1
print(income["income"].value_counts())
income['income']=income['income'].map({'<=50K': 0, '>50K': 1})

###One hot encoding 
one_hot_cols=['workclass','marital.status','occupation','native.country']
one_hot_data = pd.get_dummies(income[one_hot_cols])
print(one_hot_data.shape)
print(one_hot_data.columns.values)

##Final Data
print(income.shape)
income_final = pd.concat([income, one_hot_data], axis=1)
print(income_final.shape)
print(income_final.info())

##Features
one_hot_features=list(one_hot_data.columns.values)
numerical_features=['age',  'education.num', 'sex', 'capital.gain', 'capital.loss', 'hours.per.week']
all_features=one_hot_features+numerical_features
print(all_features)

##Data
X=income_final[all_features]
y=income_final['income']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

##Model Building
gbm_model1 = GradientBoostingClassifier(learning_rate=0.01, max_depth=4,  n_estimators=100, verbose=1)
gbm_model1.fit(X_train, y_train)

##Validation on train and test data

#Train data
predictions=gbm_model1.predict(X_train)
actuals=y_train
cm = confusion_matrix(actuals,predictions)
print("Confusion Matrix on Train data\n", cm)
accuracy=(cm[0,0]+cm[1,1])/(sum(sum(cm)))
print("Train Accuracy", accuracy)

#Test data
predictions=gbm_model1.predict(X_test)
actuals=y_test
cm = confusion_matrix(actuals,predictions)
print("Confusion Matrix on Test data\n", cm)
accuracy=(cm[0,0]+cm[1,1])/(sum(sum(cm)))
print("Test Accuracy", accuracy)

##Loop for different iterations
for i in range(5,1000, 50):
    gbm_model1 = GradientBoostingClassifier(learning_rate=0.01, max_depth=4,  n_estimators=i)
    gbm_model1.fit(X_train, y_train)
    
    print("N_estimators=" , i)
    #Train data
    predictions=gbm_model1.predict(X_train)
    actuals=y_train
    cm = confusion_matrix(actuals,predictions)
    accuracy=(cm[0,0]+cm[1,1])/(sum(sum(cm)))
    print("Train Accuracy", accuracy)

    #Test data
    predictions=gbm_model1.predict(X_test)
    actuals=y_test
    cm = confusion_matrix(actuals,predictions)
    accuracy=(cm[0,0]+cm[1,1])/(sum(sum(cm)))
    print("Test Accuracy", accuracy)

