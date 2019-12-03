#K-Means clustering implementation

#Some hints on how to start have been added to this file.
#You will have to add more code than just the hints provided here for the full implementation.


# ====
# Define a function that computes the distance between two data points
def dist(x,y):
    import numpy as np

    return np.sqrt(np.sum((x-y)**2))


# ====
# Define a function that reads data in from the csv files  HINT: http://docs.python.org/2/library/csv.html
def ReadCSV():
    #data is read from CSV using pandas csv library
    import pandas
    #df = pandas.read_csv('data1953.csv')
    #df = pandas.read_csv('data2008.csv')
    df = pandas.read_csv('dataBoth.csv')
    ##print(df)
    classes = df['LifeExpectancy']
    return df,classes


# ====
# Write the initialisation procedure
#import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
 

# ====
# Implement the k-means algorithm, using appropriate looping
df,classes = ReadCSV()
## Drop the other coloumn Countries values from data
df = df.drop(['Countries'],axis=1) 

## Convert dataframe into list and then into a numpy array
data = df.values.tolist() 
data = np.array(data)
#print(data)

## First 147 points (75% of set) are used for training and the rest is used for testing
train_data = data[:147]  
test_data = data[147:]

X = train_data
n_clusters = int(input("Please enter the number of clusters you would like to use: "))
kmeans = KMeans(n_clusters)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

labels = kmeans.predict(df)
centroids = kmeans.cluster_centers_

# ====
# Print out the results
plt.scatter(X[:,0], X[:,1], c=y_kmeans)
centers = kmeans.cluster_centers_
plt.scatter(centers[:,0], centers[:,1], c='blue');  #s=50 makes size of center of clusters =50
#plt.plot(test_data,y_kmeans, c='b')
plt.show()

