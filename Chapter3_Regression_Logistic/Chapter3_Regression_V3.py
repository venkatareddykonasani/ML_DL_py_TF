# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 13:54:00 2019

@author: Venkata Reddy Konasani
"""

import pandas as pd

emp_profile=pd.read_csv(r"D:\Google Drive\Training\Book\0.Chapters\Chapter3 Regression and Logistic\Datasets\employee_profile.csv")

#First few rows
emp_profile.head()

#Column names 
print(emp_profile.columns)


#Drawing the Scatter Plotq
import matplotlib.pyplot as plt
plt.scatter(emp_profile["Monthly_Income"], emp_profile["Monthly_Expenses"])
plt.title('Income vs Expenses Plot')
plt.xlabel('Monthly Income')
plt.ylabel('Monthly Expenses')
plt.show()

import matplotlib.pyplot as plt
plt.scatter(emp_profile["Monthly_Income"], emp_profile["Time_Spent_Reading_Books"])
plt.title('Income vs Time_Spent_Reading_Books Plot')
plt.xlabel('Monthly Income')
plt.ylabel('Time_Spent_Reading_Books')
plt.show()

#Regression Model Building
air_pass=pd.read_csv(r"D:\Google Drive\Training\Book\0.Chapters\Chapter3 Regression and Logistic\Datasets\Air_Passengers.csv")

print(air_pass.columns)

import statsmodels.formula.api as sm
model1 = sm.ols(formula='Passengers_count ~ marketing_cost', data=air_pass)
fitted1 = model1.fit()
fitted1.summary()

#prediction from the model
new_data=pd.DataFrame({"marketing_cost":[4500]})
print(fitted1.predict(new_data))

new_data1=pd.DataFrame({"marketing_cost":[4500,3600, 3000,5000]})
print(fitted1.predict(new_data1))


#Predictions for the data
air_pass["passengers_count_pred"]=round(fitted1.predict(air_pass))
keep_cols=["marketing_cost", "Passengers_count", "passengers_count_pred"]
air_pass[keep_cols]


#R-Squared Value
model2 = sm.ols(formula='Passengers_count ~ customer_ratings', data=air_pass)
fitted2 = model2.fit()
fitted2.summary()

# Multiple regression
import statsmodels.formula.api as sm
model3 = sm.ols(formula='Passengers_count ~ marketing_cost+percent_delayed_flights+number_of_trips+customer_ratings+poor_weather_index+percent_female_customers+Holiday_week+percent_male_customers', data=air_pass)
fitted3 = model3.fit()
fitted3.summary()

# Multicollinearity
income_expenses=pd.read_csv(r"D:\Google Drive\Training\Book\0.Chapters\Chapter3 Regression and Logistic\Datasets\customer_income_expenses.csv")

print(income_expenses.columns)

model4=sm.ols(formula='Monthly_Expenses ~ Monthly_Income_in_USD+Number_of_Credit_cards+Number_of_personal_loans+Monthly_Income_in_Euro', data=income_expenses)
fitted4 = model4.fit()
fitted4.summary()

#Model after dropping Monthly_Income_in_USD
model5=sm.ols(formula='Monthly_Expenses ~Number_of_Credit_cards+Number_of_personal_loans+Monthly_Income_in_Euro', data=income_expenses)
fitted5 = model5.fit()
fitted5.summary()



#VIF Function 
def vif_cal(x_vars):
    xvar_names=x_vars.columns
    for i in range(0,xvar_names.shape[0]):
        y=x_vars[xvar_names[i]] 
        x=x_vars[xvar_names.drop(xvar_names[i])]
        rsq=sm.ols(formula="y~x", data=x_vars).fit().rsquared  
        vif=round(1/(1-rsq),2)
        print (xvar_names[i], " VIF = " , vif)

#Calculating VIF values using that function
vif_cal(x_vars=income_expenses.drop(["Monthly_Expenses"], axis=1))

#Calculating VIF values after dropping Monthly_Income_in_Euro
vif_cal(x_vars=income_expenses.drop(["Monthly_Expenses","Monthly_Income_in_Euro"], axis=1))

#Calculating VIF values after dropping Monthly_Income_in_Euro and Number_of_Credit_cards
vif_cal(x_vars=income_expenses.drop(["Monthly_Expenses","Monthly_Income_in_Euro","Number_of_personal_loans"], axis=1))

#The Final model after removing all the multicollinearity 
model6=sm.ols(formula='Monthly_Expenses ~ Monthly_Income_in_USD+Number_of_Credit_cards', data=income_expenses)
fitted6 = model6.fit()
fitted6.summary()


#Calculating VIF values for airpassengers data 
vif_cal(x_vars=air_pass.drop(["Passengers_count","passengers_count_pred"], axis=1))

#Dropped percent_male_customers due to high VIF
vif_cal(x_vars=air_pass.drop(["Passengers_count","passengers_count_pred", "percent_male_customers"], axis=1))

#Dropped percent_male_customers and percent_delayed_flights due to high VIF
vif_cal(x_vars=air_pass.drop(["Passengers_count","passengers_count_pred","percent_male_customers", "percent_delayed_flights"], axis=1))

#Model after exclusing the high VIF variables
import statsmodels.formula.api as sm
model7 = sm.ols(formula='Passengers_count ~ marketing_cost+number_of_trips+customer_ratings+poor_weather_index+percent_female_customers+Holiday_week', data=air_pass)
fitted7 = model7.fit()
fitted7.summary()


#Individual impact of the variables 
##Drop two variables non-impacful number_of_trips and percent_female_customers
import statsmodels.formula.api as sm
model8 = sm.ols(formula='Passengers_count ~ marketing_cost+customer_ratings+poor_weather_index+Holiday_week', data=air_pass)
fitted8 = model8.fit()
fitted8.summary()


#Drop an impactful variable
import statsmodels.formula.api as sm
model9 = sm.ols(formula='Passengers_count ~  customer_ratings+poor_weather_index+Holiday_week', data=air_pass)
fitted9 = model9.fit()
fitted9.summary()

##################################
####Logistic Regression
#################################

#product_sales Model
product_sales=pd.read_csv(r"D:\Google Drive\Training\Book\0.Chapters\Chapter3 Regression and Logistic\Datasets\Product_sales.csv")
print(product_sales.columns)

import statsmodels.formula.api as sm
model10 = sm.ols(formula='Bought ~  Income', data=product_sales)
fitted10 = model10.fit()
fitted10.summary()


#prediction from the model
new_data=pd.DataFrame({"Income":[4000]})
print(fitted10.predict(new_data))

new_data1=pd.DataFrame({"Income":[85000]})
print(fitted10.predict(new_data1))

#product_sales data sample
print(product_sales.sample(10))

#Drawing the Scatter Plot
import matplotlib.pyplot as plt
plt.scatter(product_sales["Income"], product_sales["Bought"])
plt.title('Income vs Bought Plot')
plt.xlabel('Income')
plt.ylabel('Bought')
plt.show()

#Drawing the Regression line
pred_values= fitted10.predict(product_sales["Income"]) 
plt.scatter(product_sales["Income"], product_sales["Bought"])
plt.plot(product_sales["Income"], pred_values, color='green')
plt.title('Income vs Bought Plot')
plt.xlabel('Income')
plt.ylabel('Bought')
plt.show()

#3.9 Logistic Regression Model building 
import statsmodels.api as sm
logit_model=sm.Logit(product_sales["Bought"],product_sales["Income"])
#Model with intercept
logit_model1=sm.Logit(product_sales["Bought"],sm.add_constant(product_sales["Income"]))
logit_fit1=logit_model1.fit()
logit_fit1.summary()


#prediction from the model
new_data=pd.DataFrame({"Constant":[1,1],"Income":[4000, 85000]})
print(logit_fit1.predict(new_data))

#Drawing the Logistic line

new_data=product_sales.drop(["Bought"], axis=1)
new_data["Constant"]=1
new_data=new_data[["Constant","Income"]]
#Pass the variables to get the predicted values. Add actual values in a new column 
new_data["pred_values"]= logit_fit1.predict(new_data)
new_data["Actual"]=product_sales["Bought"]
#Sort the data and draw the graph
new_data=new_data.sort_values(["pred_values"])
plt.scatter(new_data["Income"], new_data["Actual"])
plt.plot(new_data["Income"], new_data["pred_values"], color='green')
#Add lables and title 
plt.title('Predicted vs Actual Plot')
plt.xlabel('Income')
plt.ylabel('Bought')
plt.show()


#Accuracy of the model 
print(product_sales.head(10))

#Add a new column for intercept. This will be used in prediction
product_sales["Constant"]=1
#Get the predicted values into a new column
product_sales["pred_Bought"]=logit_fit1.predict(product_sales[["Constant","Income"]])
product_sales["pred_Bought"]=round(product_sales["pred_Bought"])

#Data after updating with predicted values
print(product_sales[["Bought","pred_Bought"]])

from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(product_sales["Bought"],product_sales["pred_Bought"])
print(cm1)

accuracy1=(cm1[0,0]+cm1[1,1])/(cm1[0,0]+cm1[0,1]+cm1[1,0]+cm1[1,1])
print(accuracy1)

#3.10 Multiple Logistic Regression Line

telco_cust=pd.read_csv(r"D:\Google Drive\Training\Book\0.Chapters\Chapter3 Regression and Logistic\Datasets\telco_data.csv")
print(telco_cust.shape)
print(telco_cust.columns)

import statsmodels.api as sm
logit_model2=sm.Logit(telco_cust['Active_cust'],telco_cust[["estimated_income"]+['months_on_network']+['complaints_count']+['plan_changes_count']+['relocated_new_place']+['monthly_bill_avg']+["CSAT_Survey_Score"]+['high_talktime_flag']+['internet_time']])
logit_fit2=logit_model2.fit()
logit_fit2.summary()

#Confuson Matrix and Accuracy
telco_cust["pred_Active_cust"]=logit_fit2.predict(telco_cust.drop(["Id","Active_cust"],axis=1))
telco_cust["pred_Active_cust"]=round(telco_cust["pred_Active_cust"])

from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(telco_cust["Active_cust"],telco_cust["pred_Active_cust"])
print(cm2)

accuracy2=(cm2[0,0]+cm2[1,1])/(cm2[0,0]+cm2[0,1]+cm2[1,0]+cm2[1,1])
print(accuracy2)

#3.11 Multicollinearity in logistic regression 

import statsmodels.formula.api as sm1
def vif_cal(x_vars):
    xvar_names=x_vars.columns
    for i in range(0,xvar_names.shape[0]):
        y=x_vars[xvar_names[i]] 
        x=x_vars[xvar_names.drop(xvar_names[i])]
        rsq=sm1.ols(formula="y~x", data=x_vars).fit().rsquared  
        vif=round(1/(1-rsq),2)
        print (xvar_names[i], " VIF = " , vif)

#Calculating VIF values using that function
vif_cal(x_vars=telco_cust.drop(["Id","Active_cust","pred_Active_cust"], axis=1))

#Drop CSAT_Survey_Score
vif_cal(x_vars=telco_cust.drop(["Id","Active_cust","pred_Active_cust","CSAT_Survey_Score"], axis=1))

import statsmodels.api as sm
logit_model3=sm.Logit(telco_cust['Active_cust'],telco_cust[["estimated_income"]+['months_on_network']+['complaints_count']+['plan_changes_count']+['relocated_new_place']+['monthly_bill_avg']+['high_talktime_flag']+['internet_time']])
logit_fit3=logit_model3.fit()
logit_fit3.summary()

#Confuson Matrix and Accuracy
telco_cust["pred_Active_cust"]=logit_fit3.predict(telco_cust.drop(["Id","Active_cust","pred_Active_cust","CSAT_Survey_Score"],axis=1))
telco_cust["pred_Active_cust"]=round(telco_cust["pred_Active_cust"])

from sklearn.metrics import confusion_matrix
cm3 = confusion_matrix(telco_cust["Active_cust"],telco_cust["pred_Active_cust"])
print(cm3)

accuracy3=(cm3[0,0]+cm3[1,1])/(cm3[0,0]+cm3[0,1]+cm3[1,0]+cm3[1,1])
print(accuracy3)

#3.12 Individual impact of the variables 

#Drop estimated_income and high_talktime_flag 
import statsmodels.api as sm
logit_model4=sm.Logit(telco_cust['Active_cust'],telco_cust[['months_on_network']+['complaints_count']+['plan_changes_count']+['relocated_new_place']+['monthly_bill_avg']+['internet_time']])
logit_fit4=logit_model4.fit()
logit_fit4.summary()

#Confuson Matrix and Accuracy
telco_cust["pred_Active_cust"]=logit_fit4.predict(telco_cust.drop(["Id","Active_cust","pred_Active_cust","CSAT_Survey_Score","estimated_income","high_talktime_flag"],axis=1))
telco_cust["pred_Active_cust"]=round(telco_cust["pred_Active_cust"])

from sklearn.metrics import confusion_matrix
cm4= confusion_matrix(telco_cust["Active_cust"],telco_cust["pred_Active_cust"])
print(cm4)

accuracy4=(cm4[0,0]+cm4[1,1])/(cm4[0,0]+cm4[0,1]+cm4[1,0]+cm4[1,1])
print(accuracy4)




