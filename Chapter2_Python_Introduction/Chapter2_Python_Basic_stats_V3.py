# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 15:26:58 2019

@author: Venkata Reddy Konasani
"""
#First few lines of code 
print(601+49)
print(19*17)
print("Python code")
x=7
print(x)

#Single line comments 
"""
Multi line
Commnents 
"""

#Example of errors
#Error1
Print(600+900) #used Print() instead of print()

#Error2
print(576-'96')

#Lines will not be executed after error 
print(576-'96')
y=10
print(y)

#Variable names
1x=20 #Doesn't work

x1=20 #works

x.1=20 #Doesn't work

x_1=20 #works

#6.	Dynamic assignment line by line execution
income=12000
print(income)

#Doesn't work
z=x*y
print(z)

x=20
print(x)

y=30
z=x*y
print(z)

income="March"
print(income)

#7.	Printing along with a message. 
age=35
print("Age value is", age) 

income=5000
print("income is", income , "$")

gdp_percap=59531
Conutry="United States" 
print("GDP per capita of", Conutry, "is" , gdp_percap)



#Type of objects in python 
sales=30000
print(sales)
print("type of object is ", type(sales)) 

Avg_expenses =5000.25
print(Avg_expenses)
print("type of object is ", type(Avg_expenses)) 


##Strings in python
##Defining Strings 
name="Sheldon"
msg="sent a mail to Jack"

##Accessing strings
print(name[0])
print(name[1])

##Substring
print(name[0:4])
print(name[4:6])
print(msg[0:9])
print(msg[9:14])

##length of string
print(len(msg))
print(msg[9:len(msg)])

##String Concatenation
new_msg= name +" " +  msg
print(new_msg)

##Lists
##Defining lists
mylist1=["Sheldon","Tommy", "Benny"]
print(type(mylist1))

#Accessing values from a list
print(mylist1[0])
print(mylist1[1])

#Number of elements in the list can be accessed by using len() function 
len(mylist1)

#List appending or combining two lists into one
mylist2=["Ken","Bill"]
new_list=mylist1 + mylist2
print(new_list)

#Updating an element in the list
print("actual list",mylist1)
mylist1[0]="John"
print("list after updating" ,mylist1)

#Deleting an element from the list
print("actual list",mylist2)
del mylist2[0]
print("list after deleting" ,mylist2)

#Difference between array and a list 
val1=[1,7,6]
val2=[6,2,2]
val3=val1+val2
print(val3)

#Can this list contain different data types? 
details=["John", 1500, "LA"]
print(details)

#Can there be a list inside a list?  - YES
details_all=["John", 1500, "LA",mylist1 ]
print(details_all)

#4.	Dictionaries
#Defining a dictionary 
city={2:"Los Angeles", 9:"Dallas" , 21:"Boston"}
print(city)
print(type(city))

#Accessing values from a dictionary
print(city[9])
print(city[2])


#Printing all keys 
print(city.keys())

#Printing all values
print(city.values())

#Updating an element value in the dictionary
city[2]="New York"
print(city)

#Deleting a value from dictionary
del(city[2])
print(city)

#Can there be a  of the key? - NO 
country={1:"USA", 6:"Brazil" , 7:"India", 6: "France" }
print(country)

#Can the keys be non-numeric – YES
GDP= {"USA": 20494, "China" : 13407}
print(GDP)
print(GDP["USA"])
print(GDP[USA])#This code doesn't work

#Can the values be a list?- YES
cust={"cust1":[19, 9500], "cust2":[21, 10000]}
print(cust)
print(cust["cust1"])

#Packages
print(log(10))
print(sqrt(256))

import math
print(math.log(10))
print(math.sqrt(256))

#Numpy

import numpy as np

income = np.array([6725, 9365, 8030, 8750])
print(type(income))
print(income) 
print(income[0])

#Creating a new array from income array
expenses=income*0.65
print(expenses)

savings=income-expenses
print(savings)


#Array vs List
income_list=[6725, 9365, 8030, 8750]
income_array = np.array(income_list)

print(income_list*2)
print(income_array*2)

#Pandas
import pandas as pd
bank= pd.read_csv('D:/Google Drive/Training/Book/0.Chapters/Chapter2 Python Programming/Datasets/Bank Tele Marketing/bank_market.csv')
print(bank)

bank1= pd.read_csv('D:\\Google Drive\\Training\\Book\\0.Chapters\\Chapter2\\Datasets\\Bank Tele Marketing\\bank_market.csv')
print(bank1)

bank2= pd.read_csv(r'D:\Google Drive\Training\Book\0.Chapters\Chapter2\Datasets\Bank Tele Marketing\bank_market.csv')
print(bank2)

#Matplotlib 
bank= pd.read_csv('D:/Google Drive/Training/Book/0.Chapters/Chapter2 Python Programming/Datasets/Bank Tele Marketing/bank_market.csv')
print(bank.columns)

import matplotlib as mp

mp.pyplot.scatter(bank['age'],bank['balance'])


#If condition
#Eg-1
level=60
if level<50:
    print("Stage1")
print("Done with If")
#Eg-2
level=60
if level<50:
    print("Stage1")
    print("Done with If")
#Eg-3
level=40
if level<50:
    print("Stage1")
print("Done with If")
#Eg-4
level=60
if level<50:
    print("Stage1")
else:
    print("Stage2")
print("Done with If")

#For loop
#Eg1
names=["Tommy", "Benny", "Ken"]
for i in names:
    print("The name is", i)
#Eg2
nums=range(1,10)
cumsum=0
for i in nums:
    cumsum=cumsum+i
    print("Cumulative sum till", i ,"is", cumsum)

#2.3.1 Data Importing and basic details 
#Eg1
import pandas as pd
sales= pd.read_csv('D:\\Google Drive\\Training\\Book\\0.Chapters\\Chapter2 Python Programming\\Datasets\\Sales\\Sales.csv')
print(sales)
#Eg2
wb_data = pd.read_excel("D:\\Google Drive\\Training\\Book\\0.Chapters\\Chapter2 Python Programming\\Datasets\\World Bank Data\\World Bank Indicators.xlsx" , "Data by country")
print(wb_data)

#Basic Commands on data
print(sales.shape)
print(sales.columns)
print(sales.head(10))
print(sales.tail(10)) 
print(sales.sample(n=10))
print(sales.dtypes) 
print(sales.describe())
print(sales["Invoice_Amount"].describe())
print(sales["Sales_Type"].value_counts())
print(sum(sales["Country_code"].isnull()))
print(sum(sales["CustName"].isnull()))
print(sum(sales["Invoice_Amount"].isnull()))

#2.3.2 Subsets and Data Filter
bank= pd.read_csv('D:/Google Drive/Training/Book/0.Chapters/Chapter2 Python Programming/Datasets/Bank Tele Marketing/bank_market.csv')
print(bank.shape)
print(bank.columns)

#New dataset with selected rows 
bank1 = bank.head(5)
print(bank1)

bank2=bank.iloc[2]
print(bank2)
print(type(bank2))

index_vals=[2,9,15,25]
bank3=bank.iloc[index_vals]
print(bank3)

bank3_1=bank.iloc[[2,9,15,25]]
print(bank3_1)


#New dataset by keeping selected columns
bank4 = bank[["job", "age"]]
print(bank4.head(5))

#New dataset with selected rows and columns
bank5 = bank[["job", "age"]].iloc[0:5]
print(bank5)

#New data by excluding rows
bank6=bank.drop([0,2,4,6])
print(bank6.head(5))

#New data by excluding columns
bank7=bank.drop(["Cust_num"], axis=1)
print(bank7.head(5))

#New data by excluding columns
#This code shows an error
bank7_1=bank.drop(["Cust_num"])

#Filter conditions
#Selection with a condition on variables
#For example, selection of customers with age>40.
bank8=bank[bank['age']>40]
print(bank8.shape)
#And condition & filters
bank9=bank[(bank['age']>40) & (bank['loan']=="no")]
print(bank9.shape)
#OR condition & filters
bank10=bank[(bank['age']>40) | (bank['loan']=="no")]
print(bank10.shape)

#2.3.3 Other useful commands 
#Creating new columns 
print(bank.shape)
print(bank.columns)
bank["bal_new"]=bank["balance"]*0.9
print(bank.shape)
print(bank.columns)

#Joining
#Data impoting 
product1= pd.read_csv("D:/Google Drive/Training/Book/0.Chapters/Chapter2 Python Programming/Datasets/Orders Products/Product1_orders.csv")
print(product1.shape)
print(product1.columns)

product2= pd.read_csv("D:/Google Drive/Training/Book/0.Chapters/Chapter2 Python Programming/Datasets/Orders Products/Product2_orders.csv")
print(product2.shape)
print(product2.columns)

##Inner Join
inner_data=pd.merge(product1, product2, on='Cust_id', how='inner')
print(inner_data.shape)
###Outer Join
outer_data=pd.merge(product1, product2, on='Cust_id', how='outer')
print(outer_data.shape)
##Left outer Join
L_outer_data=pd.merge(product1, product2, on='Cust_id', how='left')
print(L_outer_data.shape)
###Right outer Join
R_outer_data=pd.merge(product1, product2, on='Cust_id', how='right')
print(R_outer_data.shape)

##Inner Join2
inner_data1=pd.merge(product1, product2, left_on='Cust_id',right_on='Cust_id', how='inner')
print(inner_data1.shape)

#2.4.1.1 Mean
import pandas as pd
income_data= pd.read_csv(r"D:/Google Drive/Training/Book/0.Chapters/Chapter2 Python Programming/Datasets/Census Income Data/Income_data.csv")
print(income_data.shape)
print(income_data.columns)

#Mean
print(income_data["capital-gain"].mean())
#Median
print(income_data["capital-gain"].median())

#Varinace and SD
bank= pd.read_csv('D:/Google Drive/Training/Book/0.Chapters/Chapter2 Python Programming/Datasets/Bank Tele Marketing/bank_market.csv')

print(bank["balance"].std())
print(bank["balance"].var())

house_loan_yes=bank[bank["housing"]=="yes"]
print(house_loan_yes["balance"].std())
print(house_loan_yes["balance"].var())

house_loan_no=bank[bank["housing"]=="no"]
print(house_loan_no["balance"].std())
print(house_loan_no["balance"].var())

#Percentiles
print(income_data["capital-gain"].quantile(0.2))

print(income_data["capital-gain"].quantile([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]))


#Percentiles deep dive
print(income_data["capital-gain"].quantile([0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99,1]))

#Hours per week
print(income_data["hours-per-week"].quantile([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]))

print(income_data["hours-per-week"].quantile([0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99,1]))

print(income_data["hours-per-week"].quantile([0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]))

##Box Plots
import matplotlib.pyplot as plt
print(income_data["age"].describe())
plt.boxplot(income_data["age"])

print(income_data["capital-gain"].describe())
plt.boxplot(income_data["capital-gain"])

print(income_data["hours-per-week"].describe())
plt.boxplot(income_data["hours-per-week"])

#Exploring discrete and categorical variables
income_data["education"].value_counts()

#Along with percentage
freq=income_data["education"].value_counts()
percent=income_data["education"].value_counts(normalize=True)
freq_table=pd.concat([freq,percent], axis=1, keys=["counts","percent"])
print(freq_table)
#Frequnecy example -2 
income_data["workclass"].value_counts()

#Cross tab
print(income_data["Income_band"].value_counts())
print(income_data["Income_band"].value_counts(normalize=True))
cross_tab=pd.crosstab(income_data["education"],income_data['Income_band'])
cross_tab_p=cross_tab.astype(float).div(cross_tab.sum(axis=1), axis=0)
final_table=pd.concat([cross_tab,cross_tab_p], axis=1)
print(final_table)
