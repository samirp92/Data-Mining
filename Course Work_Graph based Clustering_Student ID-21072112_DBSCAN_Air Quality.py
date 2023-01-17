# Import the required libraries
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

dataframe= pd.read_excel('AirQualityUCI.xlsx', header = [0])  # To read the file
print(dataframe)  # To print the data
data = dataframe.iloc[:, [3, 6, 8, 10, 11]]  # Filter the sensor data only
data.columns = ['tin oxide (CO targeted)', 'titania (NMHC targeted)', 'tungsten oxide (NOx targeted)', 
                'tungsten oxide (NO2 targeted)', 'indium oxide (O3 targeted)']  # Remane the column name 

# Checking for NULL data in the dataset
print(data.isnull().any().any())

# Printing and Storing the data
print(data)  
data.to_excel('data.xlsx')

# Visulazation of data using pairplot
sns.pairplot(data)

# Find the co-relation using heat map
plt.figure(dpi = 300)
sns.heatmap(data.corr(), annot = True, square = True)
plt.show()

features = ['tin oxide (CO targeted)', 'titania (NMHC targeted)']  # To filter the value of selected indicator only
df = data[features].copy()  # To store the value of selected indicator only

# n_neighbors = 5 as kneighbors function returns distance of point to itself 
neighbors = NearestNeighbors(n_neighbors = 5)  # creating an object of the NearestNeighbors class
nbrs = neighbors.fit(df) # fitting the data to the object
neigh_dist, neigh_ind = nbrs.kneighbors(df)  # finding the nearest neighbours


# Sorting and plot the distances between the data points
sort_neigh_dist = np.sort(neigh_dist, axis = 0)
k_dist = sort_neigh_dist[:, 4]
plt.figure(dpi = 300)
plt.plot(k_dist)
plt.axhline(y = 35, linewidth = 1, linestyle = 'dashed')
plt.ylabel('k-NN distance')
plt.xlabel('Sorted observations (4th NN)')
plt.title('k-NN distance plot', size = 12)
plt.show()

# Create the clusters with DBSCAN
clusters = DBSCAN(eps = 35, min_samples = 4).fit(df)
# get cluster labels
clusters.labels_

# Plot the data with clusters
plt.figure(dpi = 300)
p = sns.scatterplot(data = df, x = 'tin oxide (CO targeted)', y = 'titania (NMHC targeted)', 
                    hue = clusters.labels_, legend = "full", palette = "deep")
sns.move_legend(p, bbox_to_anchor = (1.0, 1.0), title = 'Clusters', loc = 'upper left')
plt.show()