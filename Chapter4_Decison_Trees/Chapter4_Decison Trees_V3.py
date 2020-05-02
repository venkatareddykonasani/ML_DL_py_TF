#Import Data
import pandas as pd
survey_data = pd.read_csv(r'D:\Google Drive\Training\Book\0.Chapters\Chapter4 Decison Trees\Datasets\Call_center_survey.csv')

#total number of customers
print(survey_data.shape)

#Column names
print(survey_data.columns)

#Print Sample data
pd.set_option('display.max_columns', None) #This option displays all the columns 

survey_data.head()

#Sample summary
summary=survey_data.describe()
round(summary,2)

#frequency counts table
survey_data['Overall_Satisfaction'].value_counts()
survey_data["Personal_loan_ind"].value_counts()
survey_data["Home_loan_ind"].value_counts()
survey_data["Prime_Customer_ind"].value_counts()


#Non numerical data need to be mapped to numerical data. 
survey_data['Overall_Satisfaction'] = survey_data['Overall_Satisfaction'].map( {'Dis Satisfied': 0, 'Satisfied': 1} ).astype(int)

#number of satisfied customers
survey_data['Overall_Satisfaction'].value_counts()

#Defining Features and lables, ignoring cust_num and target variable
features=list(survey_data.columns[1:6])
print(features)
#Preparing X and Y data
#X = survey_data[["Age", "Account_balance","Personal_loan_ind","Home_loan_ind","Prime_Customer_ind"]]
X = survey_data[features]
y = survey_data['Overall_Satisfaction']

#Building Tree Model
from sklearn import tree
DT_Model = tree.DecisionTreeClassifier(max_depth=2)
DT_Model.fit(X,y)

##Plotting the trees - Old Method

#Before drawing the graph below command on anaconda console
#pip install pydotplus 
#pip install graphviz

from IPython.display import Image
from sklearn.externals.six import StringIO

import pydotplus
dot_data = StringIO()
tree.export_graphviz(DT_Model, #Mention the model here
                     out_file = dot_data,
                     filled=True, 
                     rounded=True,
                     impurity=False,
                     feature_names = features)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())

#Rules
print(dot_data.getvalue())

##Plotting the trees - New Method

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree, export_text
plt.figure(figsize=(15,7))
plot_tree(DT_Model, filled=True, 
                     rounded=True,
                     impurity=False,
                     feature_names = features)

print(export_text(DT_Model, feature_names = features))


#LAB : Tree Validation
########################################
##########Tree Validation
#Tree Validation
predict1 = DT_Model.predict(X)
print(predict1)

from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(y, predict1)
print(cm)

total = sum(sum(cm))
#####from confusion matrix calculate accuracy
accuracy = (cm[0,0]+cm[1,1])/total
print(accuracy)


#LAB: Overfitting
#LAB: The problem of overfitting
############################################################################ 
##The problem of overfitting

import pandas as pd
overall_data = pd.read_csv(r"D:\Google Drive\Training\Book\0.Chapters\Chapter4 Decison Trees\Datasets\Customer_profile_data.csv")

##print train.info()
print(overall_data.shape)

#First few records
print(overall_data.head())

# the data have string values we need to convert them into numerical values
overall_data['Gender'] = overall_data['Gender'].map( {'Male': 1, 'Female': 0} ).astype(int)
overall_data['Bought'] = overall_data['Bought'].map({'Yes':1, 'No':0}).astype(int)

#First few records
print(overall_data.head())

#Defining features, X and Y
features = list(overall_data.columns[1:3])
print(features)

X = overall_data[features]
y = overall_data['Bought']

print(X.shape)
print(y.shape)
#Dividing X and y to train and test data parts. The function train_test_split() takes care of it. Mention the train data percentage in the parameter train_size. 
from sklearn import model_selection
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y, train_size = 0.8 , random_state=5)

print("X_train.shape", X_train.shape)
print("y_train.shape",y_train.shape)
print("X_test.shape",X_test.shape)
print("y_test.shape",y_test.shape)

##print train.info()
##print test.info()

from sklearn import tree
#training Tree Model
DT_Model1 = tree.DecisionTreeClassifier()
DT_Model1.fit(X_train,y_train)

#Plotting the trees
from IPython.display import Image
from sklearn.externals.six import StringIO
import pydotplus
dot_data = StringIO()
tree.export_graphviz(DT_Model1,
                     out_file = dot_data,
                     feature_names = features,
                     filled=True, rounded=True,
                     impurity=False)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())

#Accuracy on train data
from sklearn.metrics import confusion_matrix

predict1 = DT_Model1.predict(X_train)
cm1 = confusion_matrix(y_train,predict1)
total1 = sum(sum(cm1))
accuracy1 = (cm1[0,0]+cm1[1,1])/total1
print("Train accuracy", accuracy1)

#Accuracy on test data
predict2 = DT_Model1.predict(X_test)
cm2 = confusion_matrix(y_test,predict2)
total2 = sum(sum(cm2))
#####from confusion matrix calculate accuracy
accuracy2 = (cm2[0,0]+cm2[1,1])/total2
print("Test accuracy",accuracy2)

####LAB: Pruning
#### max_depth parameter 
DT_Model2 = tree.DecisionTreeClassifier(max_depth= 4)
DT_Model2.fit(X_train,y_train)

predict3 = DT_Model2.predict(X_train)
predict4 = DT_Model2.predict(X_test)

#Accuracy of the model on the train data
cm1 = confusion_matrix(y_train,predict3)
total1 = sum(sum(cm1))
accuracy1 = (cm1[0,0]+cm1[1,1])/total1
print("max_depth4 Train Accuracy", accuracy1)

#Accuracy of the model on the Test Data
cm2 = confusion_matrix(y_test,predict4)
total2 = sum(sum(cm2))
accuracy2 = (cm2[0,0]+cm2[1,1])/total2
print("max_depth4 Test Accuracy", accuracy2)

#### max_depth =2
DT_Model2 = tree.DecisionTreeClassifier(max_depth= 2)
DT_Model2.fit(X_train,y_train)

predict3 = DT_Model2.predict(X_train)
predict4 = DT_Model2.predict(X_test)

#Accuracy of the model on the train data
cm1 = confusion_matrix(y_train,predict3)
total1 = sum(sum(cm1))
accuracy1 = (cm1[0,0]+cm1[1,1])/total1
print("max_depth2 Train Accuracy", accuracy1)

#Accuracy of the model on the Test Data
cm2 = confusion_matrix(y_test,predict4)
total2 = sum(sum(cm2))
accuracy2 = (cm2[0,0]+cm2[1,1])/total2
print("max_depth2 Test Accuracy", accuracy2)

#### The problem of underfitting
#### max_depth =1
DT_Model2 = tree.DecisionTreeClassifier(max_depth= 1)
DT_Model2.fit(X_train,y_train)

predict3 = DT_Model2.predict(X_train)
predict4 = DT_Model2.predict(X_test)

#Accuracy of the model on the train data
cm1 = confusion_matrix(y_train,predict3)
total1 = sum(sum(cm1))
accuracy1 = (cm1[0,0]+cm1[1,1])/total1
print("max_depth1 Train Accuracy", accuracy1)

#Accuracy of the model on the Test Data
cm2 = confusion_matrix(y_test,predict4)
total2 = sum(sum(cm2))
accuracy2 = (cm2[0,0]+cm2[1,1])/total2
print("max_depth1 Test Accuracy", accuracy2)

#### max_leaf_nodes =4
DT_Model3 = tree.DecisionTreeClassifier(max_leaf_nodes= 3)
DT_Model3.fit(X_train,y_train)

predict3 = DT_Model3.predict(X_train)
predict4 = DT_Model3.predict(X_test)

#Accuracy of the model on the train data
cm1 = confusion_matrix(y_train,predict3)
total1 = sum(sum(cm1))
accuracy1 = (cm1[0,0]+cm1[1,1])/total1
print(accuracy1)

#Accuracy of the model on the Test Data
cm2 = confusion_matrix(y_test,predict4)
total2 = sum(sum(cm2))
accuracy2 = (cm2[0,0]+cm2[1,1])/total2
print(accuracy2)
