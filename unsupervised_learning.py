import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy import stats
import seaborn as sns

# Read the data into a dataframe
data = pd.read_csv('network_data.csv')

# Print the data 
print(data[:3])

# Check the basic info
data.info()

numeric_features=data[['src_bytes', 'dst_bytes', 'duration', 'count', 'serror_rate', 'rerror_rate']] 

# Z-score standardization
scaled_data = stats.zscore(numeric_features)
print(scaled_data)

# Project the data into two dimensions with PCA
pca = PCA()
pca.fit(scaled_data)
data_pca = pca.transform(scaled_data)

# Visualize the data in a scatter plot
plt.scatter(data_pca[:, 0], data_pca[:, 1])
plt.title('PCA')
plt.show()

### Agglomerative hierarchical clustering ###

# Create agglomerative clustering model with linkage='complete'
model_complete = AgglomerativeClustering(n_clusters=4, linkage='complete')

# Fit the model
model_complete.fit(scaled_data)

# Evaluate the clustering performance by using silhouette score.
print("Silhouette score for complete linkage:", silhouette_score(scaled_data, model_complete.labels_))

# Create agglomerative clustering model with linkage='average'
model_average = AgglomerativeClustering(n_clusters=4, linkage='average')

model_average.fit(scaled_data)

print("Silhouette score for average linkage:",silhouette_score(scaled_data, model_average.labels_))

# Create agglomerative clustering model with linkage='ward' and linkage='single'
model_ward = AgglomerativeClustering(n_clusters=4, linkage='ward')

model_ward.fit(scaled_data)

print("Silhouette score for ward linkage:",silhouette_score(scaled_data, model_ward.labels_))

model_single = AgglomerativeClustering(n_clusters=4, linkage='single')

model_single.fit(scaled_data)

print("Silhouette score for single linkage:",silhouette_score(scaled_data, model_single.labels_))

# Create a dendrogram using the linkage() function with the method='average' keyword argument.
average_link=linkage(scaled_data, method='average')
plt.figure()
plt.title('Dendogram for average linkage')
average_dendo = dendrogram(average_link, p=3, truncate_mode='level')
plt.show() 

# Do the same with method='complete'
complete_link=linkage(scaled_data, method='complete')
plt.figure()
plt.title('Dendogram for complete linkage')
complete_dendo = dendrogram(complete_link, p=3, truncate_mode='level')
plt.show()

### K-means clustering ###

# Perform a k-means clustering with 4 clusters
kmeans = KMeans(n_clusters=4)
kmeans.fit(scaled_data)

# Check the silhouette score
print("Silhouette score for k-means clustering with 4 clusters:",silhouette_score(scaled_data, kmeans.labels_))

# Try a different number of clusters
kmeans = KMeans(n_clusters=8)
kmeans.fit(scaled_data)

# Check the silhouette score
print("Silhouette score for k-means clustering with 8 clusters:",silhouette_score(scaled_data, kmeans.labels_))

# The best silhouette score was aqcuired with 8 clusters.

### Comparing clusters with the true labels ###

# The best silhouette score was acquired with single linkage, so let's use that in the agglomerative hierarchical clustering  
model_single = AgglomerativeClustering(n_clusters=4, linkage='single')

model_single.fit(scaled_data)

# Next we'll visualize the data using PCA

# Create a PCA object
pca = PCA(n_components=2)

# Fit the PCA object to the data and transform the data using the PCA object
scaled_data_pca = pca.fit_transform(scaled_data)

# Create a dataframe to use in plotting the data with the right class labels
scaled_data_pca_df=pd.DataFrame(data = scaled_data_pca, columns = ['principal component 1', 'principal component 2'])

pca_dataframe = pd.concat([scaled_data_pca_df, data[['class']]], axis = 1)

# Create a scatter plot of the first two principal components with the real class values (code based on tutorial
# at https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60)
fig, ax = plt.subplots(1, 2, figsize=(12,8))
targets = ['denial_of_service', 'normal', 'probe', 'unauthorized_remote_access']
colors = ['red', 'green', 'blue','purple']
for target, color in zip(targets,colors):
    class_labels = pca_dataframe['class']  == target
    ax[0].scatter(pca_dataframe.loc[class_labels, 'principal component 1']
               , pca_dataframe.loc[class_labels, 'principal component 2']
               , c = color
               , s = 50)
ax[0].set_xlabel('Principal component 1', fontsize=12)
ax[0].set_ylabel('Principal component 2', fontsize=12)
ax[0].set_title('PCA with the real class labels', fontsize = 15)
ax[0].legend(targets)

# Create a scatter plot of the first two principal components with the predicted labels
for target, color in zip([0,1,2,3],colors):
    class_labels = model_single.labels_ == target
    ax[1].scatter(pca_dataframe.loc[class_labels, 'principal component 1']
               , pca_dataframe.loc[class_labels, 'principal component 2']
               , c = color
               , s = 50)
ax[1].set_xlabel('Principal component 1', fontsize=12)
ax[1].set_ylabel('Principal component 2', fontsize=12)
ax[1].set_title('PCA with the predicted class labels', fontsize = 15)
ax[1].legend(targets)

# Check the adjusted Rand score
print("Adjusted Rand score:", adjusted_rand_score(pca_dataframe['class'], model_single.labels_))

# The clusters match with each other very poorly, as it's seen from both the plots and from the adjusted Rand score. 
# The score is very low, which means that the similarity between the clusters is almost nonexistent

# Perform a k-means clustering
kmeans = KMeans(n_clusters=4)
kmeans.fit(scaled_data)

# Create a PCA object
pca_kmeans = PCA(n_components=2)

# Fit the PCA object to the data and transform the data using the PCA object
scaled_data_pca_kmeans = pca_kmeans.fit_transform(scaled_data)

# Create a scatter plot of the first two principal components with the real class values
fig1, ax1 = plt.subplots(1, 2, figsize=(12,8))
for target, color in zip(targets,colors):
    class_labels = pca_dataframe['class']  == target
    ax1[0].scatter(pca_dataframe.loc[class_labels, 'principal component 1']
               , pca_dataframe.loc[class_labels, 'principal component 2']
               , c = color
               , s = 50)
ax1[0].set_xlabel('Principal component 1', fontsize=12)
ax1[0].set_ylabel('Principal component 2', fontsize=12)
ax1[0].set_title('PCA with the real class labels', fontsize = 15)
ax1[0].legend(targets)

# Create a scatter plot of the first two principal components with the predicted labels
for target, color in zip([0,1,2,3],colors):
    class_labels1 = kmeans.labels_ == target
    ax1[1].scatter(pca_dataframe.loc[class_labels1, 'principal component 1']
               , pca_dataframe.loc[class_labels1, 'principal component 2']
               , c = color
               , s = 50)
ax1[1].set_xlabel('Principal component 1',fontsize=12)
ax1[1].set_ylabel('Principal component 2', fontsize=12)
ax1[1].set_title('PCA with the predicted class labels', fontsize = 15)
ax1[1].legend(targets)

# Check the adjusted Rand score
print("Adjusted Rand score:", adjusted_rand_score(pca_dataframe['class'], kmeans.labels_))

# The adjusted Rand score is higher with the k-means clustered data, so according to the adjusted Rand score the k-means clustering performed better.

### Clustering unlabeled data ###

# Read the data into a dataframe
data_seeds = pd.read_csv('seeds_data.csv')

# Perform the z-score standardization
scaled_data_seeds = stats.zscore(data_seeds)
print(scaled_data_seeds)

# Project the data to two dimensions with PCA
pca_seeds = PCA()
pca_seeds.fit(scaled_data_seeds)
data_pca_seeds = pca_seeds.transform(scaled_data_seeds)

# Visualize the two-dimensional data in a scatter plot
plt.scatter(data_pca_seeds[:, 0], data_pca_seeds[:, 1])
plt.title('PCA')
plt.show()

# First we cluster the data with k-means clustering and try different values for the number of clusters
kmeans_seeds = KMeans(n_clusters=2)
kmeans_seeds.fit(scaled_data_seeds)

# Check the silhouette score
print("Silhouette score for k-means clustering with 2 clusters:",silhouette_score(scaled_data_seeds, kmeans_seeds.labels_))

# Next we'll perform agglomerative hierarchical clustering with different linkage criteria
agg_seeds = AgglomerativeClustering(n_clusters=2, linkage='ward')

# Fit the model
agg_seeds.fit(scaled_data_seeds)

# Evaluate the clustering performance by using silhouette score.
print("Silhouette score for complete linkage:", silhouette_score(scaled_data_seeds, agg_seeds.labels_))

# Visualize the clusters in PCA
plt.scatter(data_pca_seeds[:, 0], data_pca_seeds[:, 1], c=kmeans_seeds.labels_)
plt.xlabel('Principal component 1',fontsize=12)
plt.ylabel('Principal component 2', fontsize=12)
plt.title('PCA with the predicted class labels', fontsize = 15)

# The best-performing result separates the data from the middle into two clusters. 
# It's difficult to say for sure if it is correct without knowing the true class labels, 
# but if the true amount of clusters is two, then it makes sense that the clusters separate from the middle of the plot.
