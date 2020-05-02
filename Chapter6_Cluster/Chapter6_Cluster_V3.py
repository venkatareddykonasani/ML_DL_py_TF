import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the wholesale customers dataset
cust_data = pd.read_csv(r"D:\Google Drive\Training\Book\0.Chapters\Chapter6 Cluster Analysis\Datasets\Wholesale\Wholesale_customers_data.csv")

#Rows and Columns
print(cust_data.shape)
print(cust_data.columns.values)
#Dataset Information
cust_data.info()

#Head
pd.set_option('display.max_columns', None) #This option displays all the columns 
cust_data.sample(n=5, random_state=77)

#Frequency Counts
cust_data["Channel"].value_counts()
cust_data["Region"].value_counts()

#Summary
round(cust_data.describe(),2)

#Box Plots
plt.figure(figsize=(10,10))
plt.title("All the variables box plots", size=20)
sns.boxplot(x="variable", y="value", data=pd.melt(cust_data[['Fresh', 'Milk', 'Grocery','Frozen', 'Detergents_Paper', 'Delicatessen']]))
plt.show()

#Distance Measure
cust_data_sample=cust_data.sample(n=5, random_state=77)
cust_data_sample[["Cust_id","Fresh", "Grocery"]]

#Scatter plot of customers
plt.figure(figsize=(10,10))
plt.title("Fresh and Grocery spending plot", size=20)
plot=sns.scatterplot(x="Fresh",y="Grocery", data=cust_data_sample, s=500)
for i in list(cust_data_sample.index):
    plot.text(cust_data_sample.Fresh[i],cust_data_sample.Grocery[i],cust_data_sample.Cust_id[i],size=20)

#############
## Distance Matrix

def distance_cal(data_frame):
    distance_matrix=np.zeros((data_frame.shape[0],data_frame.shape[0]))
    for i in range(0 , data_frame.shape[0]):
        for j in range(0 , data_frame.shape[0]):
            distance_matrix[i,j]=round(np.sqrt(sum((data_frame.iloc[i] - data_frame.iloc[j])**2)))
    return(distance_matrix)

distance_matrix=distance_cal(cust_data_sample[["Fresh", "Grocery"]])
print(distance_matrix)
print(distance_matrix[0,0])
print(distance_matrix[1,0])
print(distance_matrix[2,1])

####


############################
## Building clusters in Python    
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, random_state=333) # Mention the Number of clusters
X=cust_data.drop(['Cust_id', 'Channel', 'Region'],axis=1) # Custid is not needed
kmeans = kmeans.fit(X) #Model building

# Getting the cluster labels and attaching them to the original data
cust_data_clusters=cust_data
cust_data_clusters["Cluster_id"]= kmeans.predict(X)
cust_data_clusters.head(10)

#Final Results
cluster_counts=cust_data_clusters['Cluster_id'].value_counts(sort=False)
cluster_means= cust_data_clusters.groupby(['Cluster_id']).mean()

print(cluster_counts)
print(round(cluster_means,1))

##Cluster wise Spendings box plot
df_melt = pd.melt(cust_data_clusters.drop(['Cust_id', 'Channel', 'Region'],axis=1), "Cluster_id", var_name="Prod_type", value_name="Spend")

plt.figure(figsize=(20,10))
sns.boxplot(x='Cluster_id', hue="Prod_type", y="Spend", data=df_melt)
plt.title("Cluster wise Spendings", size=20)

##Cluster 0 to 4
Cluster=df_melt[(df_melt['Cluster_id']==4)] #Change this from 0  to 4 
plt.figure(figsize=(7,7))
sns.boxplot(x='Cluster_id', hue="Prod_type", y="Spend", data=Cluster)
plt.title("Cluster4", size=20)


##Objective 1
Cluster_2and3=df_melt[(df_melt['Cluster_id']==2)| (df_melt['Cluster_id']==3)]
plt.figure(figsize=(10,7))
sns.boxplot(x='Cluster_id', hue="Prod_type", y="Spend", data=Cluster_2and3)
plt.title("Cluster 2 and 3 only", size=20)

obj1_data= cust_data_clusters[(cust_data_clusters['Cluster_id']==2)| (cust_data_clusters['Cluster_id']==3)]
print(list(obj1_data["Cust_id"]))

##Objective 2
obj2_data= cust_data_clusters[cust_data_clusters['Cluster_id']==4]
print(list(obj2_data["Cust_id"]))

##Objective 3
obj3_data= cust_data_clusters[cust_data_clusters['Cluster_id']==1]
print(list(obj3_data["Cust_id"]))

##Inertia
print(kmeans.inertia_)

###########################
####Elbow Method
elbow_data=pd.DataFrame()
for i in range(1,16):
    kmeans_m2 = KMeans(n_clusters=i, random_state=333) # Mention the Number of clusters
    X=cust_data.drop(['Cust_id', 'Channel', 'Region'],axis=1) # Custid is not needed
    model= kmeans_m2.fit(X)
    elbow_data.at[i,"K"]=i
    elbow_data.at[i,"Inertia"]=round(model.inertia_)/10000000 #To lower the values
print(elbow_data)


##Elbow Plot
plt.figure(figsize=(15,8))
plt.title("Elbow Plot", size=20)
plt.plot(elbow_data["K"],elbow_data["Inertia"],'--bD')
plt.xticks(elbow_data["K"])
plt.xlabel("K", size=15)
plt.ylabel("Inertia", size=15)
