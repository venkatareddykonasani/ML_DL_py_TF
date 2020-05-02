"""
@author: Venkata Reddy Konasani
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix as cm
import neurolab as nl
import random

####################Decison Boundry######################
###Data
Data_path="D:\\Google Drive\\Training\\Book\\0.Chapters\\Chapter8 ANN\\Datasets"

Emp_Purchase_raw = pd.read_csv(Data_path+"\\Emp_Purchase\\Emp_Purchase.csv")

####Filter the data and take a subset from above dataset . Filter condition is Sample_Set<3
Emp_Purchase1=Emp_Purchase_raw[Emp_Purchase_raw.Sample_Set<3]
print(Emp_Purchase1.shape)
print(Emp_Purchase1.columns.values)
print(Emp_Purchase1.head(10))

#frequency table of Purchase variable
Emp_Purchase1.Purchase.value_counts()

####The clasification graph
#Draw a scatter plot that shows Age on X axis and Experience on Y-axis. Try to distinguish the two classes with colors or shapes.

fig = plt.figure()
ax1 = fig.add_subplot(111)
plt.rcParams["figure.figsize"] = (8,6)
plt.title('Age, Experience  vs Purchase', fontsize=20)

ax1.scatter(Emp_Purchase1.Age[Emp_Purchase1.Purchase==0],Emp_Purchase1.Experience[Emp_Purchase1.Purchase==0], s=100, c='b', marker="o", label='Purchase 0')
ax1.scatter(Emp_Purchase1.Age[Emp_Purchase1.Purchase==1],Emp_Purchase1.Experience[Emp_Purchase1.Purchase==1], s=100, c='r', marker="x", label='Purchase 1')
ax1.set_xlabel('Age',fontsize=15)
ax1.set_ylabel('Experience',fontsize=15)

plt.xlim(min(Emp_Purchase1.Age), max(Emp_Purchase1.Age))
plt.ylim(min(Emp_Purchase1.Experience), max(Emp_Purchase1.Experience))
plt.legend(loc='upper left');

plt.show()

###Logistic Regerssion model1
import statsmodels.formula.api as sm
model1 = sm.logit(formula='Purchase ~ Age+Experience', data=Emp_Purchase1)
fitted1 = model1.fit()
fitted1.summary()

#######Accuracy and error of the model1
#Create the confusion matrix
predicted_values=fitted1.predict(Emp_Purchase1[["Age"]+["Experience"]])
predicted_values[1:10]
threshold=0.5

import numpy as np
predicted_class=np.zeros(predicted_values.shape)
predicted_class[predicted_values>threshold]=1

predicted_class

from sklearn.metrics import confusion_matrix as cm
ConfusionMatrix = cm(Emp_Purchase1[['Purchase']],predicted_class)
print(ConfusionMatrix)
accuracy=(ConfusionMatrix[0,0]+ConfusionMatrix[1,1])/sum(sum(ConfusionMatrix))
print('Accuracy : ',accuracy)
error=1-accuracy
print('Error: ',error)


#coefficients
slope1=fitted1.params[1]/(-fitted1.params[2])
intercept1=fitted1.params[0]/(-fitted1.params[2])

#Finally draw the decision boundary for this logistic regression model
      
fig = plt.figure()
ax1 = fig.add_subplot(111)
plt.rcParams["figure.figsize"] = (8,6)
plt.title('Decision Boundary', fontsize=20)

ax1.scatter(Emp_Purchase1.Age[Emp_Purchase1.Purchase==0],Emp_Purchase1.Experience[Emp_Purchase1.Purchase==0], s=100, c='b', marker="o", label='Purchase 0')
ax1.scatter(Emp_Purchase1.Age[Emp_Purchase1.Purchase==1],Emp_Purchase1.Experience[Emp_Purchase1.Purchase==1], s=100, c='r', marker="x", label='Purchase 1')
ax1.set_xlabel('Age',fontsize=15)
ax1.set_ylabel('Experience',fontsize=15)

plt.xlim(min(Emp_Purchase1.Age), max(Emp_Purchase1.Age))
plt.ylim(min(Emp_Purchase1.Experience), max(Emp_Purchase1.Experience))
plt.legend(loc='upper left');

x_min, x_max = ax1.get_xlim()
ax1.plot([0, x_max], [intercept1, x_max*slope1+intercept1])

plt.show()

############################################
####Overall Data 
############################################

##plotting the overall data

fig = plt.figure()
ax1 = fig.add_subplot(111)
plt.rcParams["figure.figsize"] = (8,6)
plt.title('Age, Experience  vs Purchase - Overall Data', fontsize=20)


ax1.scatter(Emp_Purchase_raw.Age[Emp_Purchase_raw.Purchase==0],Emp_Purchase_raw.Experience[Emp_Purchase_raw.Purchase==0], s=100, c='b', marker="o", label='Purchase 0')
ax1.scatter(Emp_Purchase_raw.Age[Emp_Purchase_raw.Purchase==1],Emp_Purchase_raw.Experience[Emp_Purchase_raw.Purchase==1], s=100, c='r', marker="x", label='Purchase 1')
ax1.set_xlabel('Age',fontsize=15)
ax1.set_ylabel('Experience',fontsize=15)

plt.xlim(min(Emp_Purchase_raw.Age), max(Emp_Purchase_raw.Age))
plt.ylim(min(Emp_Purchase_raw.Experience), max(Emp_Purchase_raw.Experience))
plt.legend(loc='upper left');
plt.show()

###Logistic Regerssion model1
model = sm.logit(formula='Purchase ~ Age+Experience', data=Emp_Purchase_raw)
fitted = model.fit()
fitted.summary2()

# getting slope and intercept of the line
slope=fitted.params[1]/(-fitted.params[2])
intercept=fitted.params[0]/(-fitted.params[2])

##Accuracy and error of the model1
#Create the confusion matrix
#predicting values
predicted_values=fitted.predict(Emp_Purchase_raw[["Age"]+["Experience"]])
predicted_values[1:10]

#Lets convert them to classes using a threshold
threshold=0.5
threshold

import numpy as np
predicted_class=np.zeros(predicted_values.shape)
predicted_class[predicted_values>threshold]=1

#Predcited Classes
predicted_class[1:10]

from sklearn.metrics import confusion_matrix as cm
ConfusionMatrix = cm(Emp_Purchase_raw[['Purchase']],predicted_class)
print(ConfusionMatrix)
accuracy=(ConfusionMatrix[0,0]+ConfusionMatrix[1,1])/sum(sum(ConfusionMatrix))
print(accuracy)

error=1-accuracy
error

#Finally draw the decision boundary for this logistic regression model
fig = plt.figure()
ax1 = fig.add_subplot(111)
plt.rcParams["figure.figsize"] = (8,6)
plt.title('Decision Boundary - Overall Data', fontsize=20)

ax1.scatter(Emp_Purchase_raw.Age[Emp_Purchase_raw.Purchase==0],Emp_Purchase_raw.Experience[Emp_Purchase_raw.Purchase==0], s=100, c='b', marker="o", label='Purchase 0')
ax1.scatter(Emp_Purchase_raw.Age[Emp_Purchase_raw.Purchase==1],Emp_Purchase_raw.Experience[Emp_Purchase_raw.Purchase==1], s=100, c='r', marker="x", label='Purchase 1')
plt.xlim(min(Emp_Purchase_raw.Age), max(Emp_Purchase_raw.Age))
plt.ylim(min(Emp_Purchase_raw.Experience), max(Emp_Purchase_raw.Experience))
plt.legend(loc='upper left');
ax1.set_xlabel('Age',fontsize=15)
ax1.set_ylabel('Experience',fontsize=15)

x_min, x_max = ax1.get_xlim()
ax1.plot([0, x_max], [intercept, x_max*slope+intercept],linewidth=5, c='r')
plt.show()


############################################
####    Intermediate output Models
############################################

## h1 model
Emp_Purchase1=Emp_Purchase_raw[Emp_Purchase_raw.Sample_Set<3]
model1 = sm.logit(formula='Purchase ~ Age+Experience', data=Emp_Purchase1)
fitted1 = model1.fit()

##Predictions 
Emp_Purchase_raw['h1']=fitted1.predict(Emp_Purchase_raw[["Age"]+["Experience"]])

## h2 model 
Emp_Purchase2=Emp_Purchase_raw[Emp_Purchase_raw.Sample_Set>1]
model2 = sm.logit(formula='Purchase ~ Age+Experience', data=Emp_Purchase2)
fitted2 = model2.fit(method="bfgs")

##Predictions 
Emp_Purchase_raw['h2']=fitted2.predict(Emp_Purchase_raw[["Age"]+["Experience"]])

##h1 and h2 in the data
print(Emp_Purchase_raw[['Age', 'Experience','h1','h2','Purchase']])

##plotting the h1, h2 vs target

fig = plt.figure()
ax = fig.add_subplot(111)
plt.rcParams["figure.figsize"] = (8,6)
plt.title('h1, h2 vs target ', fontsize=20)

ax.scatter(Emp_Purchase_raw.h1[Emp_Purchase_raw.Purchase==0],Emp_Purchase_raw.h2[Emp_Purchase_raw.Purchase==0], s=100, c='b', marker="o", label='Purchase 0')
ax.scatter(Emp_Purchase_raw.h1[Emp_Purchase_raw.Purchase==1],Emp_Purchase_raw.h2[Emp_Purchase_raw.Purchase==1], s=100, c='r', marker="x", label='Purchase 1')
ax.set_xlabel('h1',fontsize=15)
ax.set_ylabel('h2',fontsize=15)

plt.xlim(min(Emp_Purchase_raw.h1), max(Emp_Purchase_raw.h1)+0.2)
plt.ylim(min(Emp_Purchase_raw.h2), max(Emp_Purchase_raw.h2)+0.2)

plt.legend(loc='lower left');
plt.show()

###Logistic Regerssion model with Intermediate outputs as input
model_combined = sm.logit(formula='Purchase ~ h1+h2', data=Emp_Purchase_raw)
fitted_combined = model_combined.fit(method="bfgs")
fitted_combined.summary()

# getting slope and intercept of the line
slope_combined=fitted_combined.params[1]/(-fitted_combined.params[2])
intercept_combined=fitted_combined.params[0]/(-fitted_combined.params[2])

##Finally draw the decision boundary for this logistic regression model
fig = plt.figure()
ax2 = fig.add_subplot(111)
plt.rcParams["figure.figsize"] = (8,7)
plt.title('h1, h2 vs target ', fontsize=20)

ax2.scatter(Emp_Purchase_raw.h1[Emp_Purchase_raw.Purchase==0],Emp_Purchase_raw.h2[Emp_Purchase_raw.Purchase==0], s=100, c='b', marker="o", label='Purchase 0')
ax2.scatter(Emp_Purchase_raw.h1[Emp_Purchase_raw.Purchase==1],Emp_Purchase_raw.h2[Emp_Purchase_raw.Purchase==1], s=100, c='r', marker="x", label='Purchase 1')
ax2.set_xlabel('h1',fontsize=15)
ax2.set_ylabel('h2',fontsize=15)

plt.xlim(min(Emp_Purchase_raw.h1), max(Emp_Purchase_raw.h1)+0.2)
plt.ylim(min(Emp_Purchase_raw.h2), max(Emp_Purchase_raw.h2)+0.2)

plt.legend(loc='lower left');

x_min, x_max = ax2.get_xlim()
y_min,y_max=ax2.get_ylim()
ax2.plot([x_min, x_max], [x_min*slope_combined+intercept_combined, x_max*slope_combined+intercept_combined],linewidth=4)
plt.show()

#######Accuracy and error of the model
#Create the confusion matrix
#Predciting Values
predicted_values=fitted_combined.predict(Emp_Purchase_raw[["h1"]+["h2"]])
predicted_values[1:10]

#Lets convert them to classes using a threshold
threshold=0.5
threshold

import numpy as np
predicted_class=np.zeros(predicted_values.shape)
predicted_class[predicted_values>threshold]=1

#ConfusionMatrix
from sklearn.metrics import confusion_matrix as cm
ConfusionMatrix = cm(Emp_Purchase_raw[['Purchase']],predicted_class)
print("ConfusionMatrix\n", ConfusionMatrix)
accuracy=(ConfusionMatrix[0,0]+ConfusionMatrix[1,1])/sum(sum(ConfusionMatrix))
print("accuracy\n", accuracy)

##the two decison boundries
slope1=fitted1.params[1]/(-fitted1.params[2])
intercept1=fitted1.params[0]/(-fitted1.params[2])

slope2=fitted2.params[1]/(-fitted2.params[2])
intercept2=fitted2.params[0]/(-fitted2.params[2])

fig = plt.figure()
ax1 = fig.add_subplot(111)
plt.rcParams["figure.figsize"] = (8,6)
plt.title('Age, Experience  vs Purchase - Overall Data', fontsize=20)


ax1.scatter(Emp_Purchase_raw.Age[Emp_Purchase_raw.Purchase==0],Emp_Purchase_raw.Experience[Emp_Purchase_raw.Purchase==0], s=100, c='b', marker="o", label='Purchase 0')
ax1.scatter(Emp_Purchase_raw.Age[Emp_Purchase_raw.Purchase==1],Emp_Purchase_raw.Experience[Emp_Purchase_raw.Purchase==1], s=100, c='r', marker="x", label='Purchase 1')
ax1.set_xlabel('Age',fontsize=15)
ax1.set_ylabel('Experience',fontsize=15)

plt.xlim(min(Emp_Purchase_raw.Age), max(Emp_Purchase_raw.Age))
plt.ylim(min(Emp_Purchase_raw.Experience), max(Emp_Purchase_raw.Experience))

x_min, x_max = ax1.get_xlim()
ax1.plot([0, x_max], [intercept1, x_max*slope1+intercept1],linewidth=4)
ax1.plot([0, x_max], [intercept2, x_max*slope2+intercept2],linewidth=4)

plt.legend(loc='upper left');
plt.show()


###############################
##Gradinet Descent for liner regression
def lr_gd(X, y, w1, w0, learning_rate, epochs):
     for i in range(epochs):
          y_pred = (w1 * X) + w0
          error = sum([k**2 for k in (y-y_pred)])
          
          ##Gradients
          w0_gradient = -sum(y - y_pred)
          w1_gradient = -sum(X * (y - y_pred))
          
          ##Weight Updating
          w0 = w0 - (learning_rate * w0_gradient)
          w1 = w1 - (learning_rate * w1_gradient)
          
          print("epoch", i, "error =>", round(error,2), "w0 => ", round(w0,2), "w1 => ",round(w1,2))
     return error, w0, w1

##Using the GD function
x_data=np.random.random(10)
y_data= x_data*20 + 10 

w0_init=5
w1_init=10

lr_gd(X=x_data, y=y_data, w1=w1_init, w0=w0_init, learning_rate=0.01, epochs=600)	 

####################################################
########Case Study – Recognizing Handwritten Digits 
##################################################

####Image importing

x=plt.imread(Data_path+"\\Sample_images\\Marketvegetables.jpg")

plt.rcParams["figure.figsize"] = (12,8)
plt.imshow(x)

print('Shape of the image',x.shape) 
print(x)

##Data Importing
#Importing test and training data

digits_data_raw = np.loadtxt(Data_path+"\\USPS\\USPS_train.txt")

##Input data is in nparry format. we convert it into dataframe for better handling
digits_data=pd.DataFrame(digits_data_raw)

#Shape of the data
print(digits_data.shape)

##Details of the data
print(digits_data.head())

##Frequency count of target
print(digits_data[0:][0].value_counts())


#Lets see some images.
#first image
i=0
data_row=digits_data_raw[i][1:]
pixels = np.matrix(data_row)
pixels=pixels.reshape(16,16)
plt.title(["Row number ", i] , fontsize=20)
plt.imshow(pixels, cmap='Greys')

#second image
i=1
data_row=digits_data_raw[i][1:]
pixels = np.matrix(data_row)
pixels=pixels.reshape(16,16)
plt.title(["Row number ", i] , fontsize=20)
plt.imshow(pixels, cmap='Greys')

#Few more images try i=100, 350, 1000, 5500
i=5000 
data_row=digits_data_raw[i][1:]
pixels = np.matrix(data_row)
pixels=pixels.reshape(16,16)
plt.title(["Row number ", i] , fontsize=20)
plt.imshow(pixels, cmap='Greys')

##Train and Test data creation
X=digits_data.drop(digits_data.columns[[0]], axis=1)
y=digits_data[0:][0]

X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size=0.2, random_state=33)

#Shape of the data
print("X_train shape", X_train.shape)
print("y_train shape", y_train.shape)
print("X_test shape", X_test.shape)
print("y_test shape", y_test.shape)


#Creating multiple binary columns for multiple outputs
#####We need these variables while building the model
digit_labels=pd.DataFrame()

#Convert target into onehot encoding
digit_labels = pd.get_dummies(y_train)

#see our newly created labels data
digit_labels.head(10)

#########Neural network building

#getting minimum and maximum of each column of x_train into a list
min_max_all_cols=[[X_train[i][0:].min(), X_train[i][0:].max()] for i in range(1,X_train.shape[1]+1)]

print(len(min_max_all_cols))
print(min_max_all_cols)

##Configure the network

net = nl.net.newff(minmax=min_max_all_cols,size=[20,10],transf=[nl.trans.LogSig()]*2)

#Training method is Resilient Backpropagation method
net.trainf = nl.train.train_rprop 

##Train the network
net.train(X_train, digit_labels, show=1, epochs=300)

#Model results
## Input to Hidden layer weights 
print(net.layers[0].np['w'])
print(net.layers[0].np['b'])

## Hidden to Output layer weights 
print(net.layers[1].np['w'])
print(net.layers[1].np['b'])

##Shape of the weights
print(net.layers[0].np['w'].shape)
print(net.layers[0].np['b'].shape)
print(net.layers[1].np['w'].shape)
print(net.layers[1].np['b'].shape)

# Prediction on test data
predicted_values = net.sim(X_test)
predicted=pd.DataFrame(predicted_values)
print(round(predicted.head(10),3))

#Converting predicted probabilities into numbers
predicted_number=predicted.idxmax(axis=1)
print(predicted_number.head(15))

##Accuracy calculation on test data
#confusion matrix
ConfusionMatrix = cm(y_test,predicted_number)
print("ConfusionMatrix on test data \n", ConfusionMatrix)

#accuracy
accuracy=np.trace(ConfusionMatrix)/sum(sum(ConfusionMatrix))
print("Test Accuracy", accuracy)

###Preictions on Random data

#i is a random number between 0 and 7291
i=623

random_sampel_data=digits_data_raw[[i]]
random_sampel_data1=pd.DataFrame(random_sampel_data)
X_sample=random_sampel_data1.drop(random_sampel_data1.columns[[0]], axis=1)

predicted_values = net.sim(X_sample)
predicted=pd.DataFrame(predicted_values)
predicted_number=predicted.idxmax(axis=1)
predicted_number

data_row=random_sampel_data[0][1:]
pixels = np.matrix(data_row)
pixels=pixels.reshape(16,16)
plt.rcParams["figure.figsize"] = (7,5)
plt.title(["Row = ", i, "Prediction Digit ", predicted_number[0]], fontsize=20)
plt.imshow(pixels, cmap='Greys')
        
