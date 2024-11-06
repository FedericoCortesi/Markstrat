import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler

class HierarchicalClustering:
    def __init__(self, data: np.ndarray, n_clusters: int = 4, linkage_method: str = 'ward', scale_data: bool = True):
        """
        Initialize the HierarchicalClustering class with the dataset and parameters.
        
        Parameters:
        - data: 2D array or pandas DataFrame (samples x features)
        - n_clusters: Number of clusters to form
        - linkage_method: Method used to calculate the distance between clusters. Default is 'ward'
        - scale_data: Whether to scale the data before clustering. Default is True
        """
        self.data = data
        self.n_clusters = n_clusters
        self.linkage_method = linkage_method
        self.scale_data = scale_data
        
        if self.scale_data:
            # Standardize the data if required
            self.scaler = StandardScaler()
            self.data_scaled = self.scaler.fit_transform(data)
        else:
            self.data_scaled = data

        # Perform clustering
        self.clustering_model = AgglomerativeClustering(n_clusters=self.n_clusters, linkage=self.linkage_method)
        self.clusters = self.clustering_model.fit_predict(self.data_scaled)
        
        self.reduced_data = self.perform_PCA(self.data_scaled)[0]

    def perform_PCA(self, data):
        # If the input is a numpy array, assign default names like 'Feature_1', 'Feature_2', ...
        feature_names = [f'Feature_{i+1}' for i in range(data.shape[1])]
    
        # Perform PCA 
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(data)
        
        # Get the PCA loadings (principal components coefficients)
        loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(pca.components_.shape[0])], index=feature_names)

        return reduced_data, loadings


    def plot_dendrogram(self):
        """
        Plot the dendrogram to visualize how the clusters are formed.
        """
        linked = linkage(self.data_scaled, method=self.linkage_method)
        
        plt.figure(figsize=(20, 7))
        dendrogram(linked, labels=self.data.index.tolist())
        plt.title("Dendrogram for Hierarchical Clustering")
        plt.xlabel("Sample index")
        plt.ylabel("Distance")
        plt.show()

    def plot_clusters(self):
        """
        Plot the clusters in the 2D PCA-reduced space.
        """
        plt.figure(figsize=(10, 7))
        plt.scatter(self.reduced_data[:, 0], self.reduced_data[:, 1], c=self.clusters, cmap='viridis')
        
        # Add labels for each data point
        for i, label in enumerate(self.data.index):
            plt.text(self.reduced_data[i, 0], self.reduced_data[i, 1], label, fontsize=7, ha='right', va='bottom')

        # Get the centroids of the clusters
        centroids = self.get_cluster_centroids()[1]
        
        # Plot the centroids with a distinct marker (e.g., red 'X')
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='o', s=25, label='Centroids')

        # Add the cluster numbers next to the centroids
        for i, (x, y) in enumerate(zip(centroids[:, 0], centroids[:, 1])):
            plt.text(x, y, f'Cluster {i}', fontsize=10, ha='center', va='center', color='black')
    
    
        plt.title(f"Agglomerative Clustering (n_clusters={self.n_clusters}) - 2D PCA")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.colorbar(label='Cluster')
        plt.show()

    def get_cluster_centroids(self):
        """
        Get the centroids of the clusters by calculating the mean of each feature for each cluster.
        
        Returns:
        - centroids: A DataFrame containing the centroids of the clusters.
        """
        cluster_centroids = pd.DataFrame(columns=[f'Feature_{i+1}' for i in range(self.data.shape[1])])
        
        for i in range(self.n_clusters):
            cluster_data = self.data_scaled[self.clusters == i]
            centroid = np.mean(cluster_data, axis=0)
            cluster_centroids.loc[i] = centroid

        pca_cluster_centroids = self.perform_PCA(cluster_centroids)
        
        return cluster_centroids, pca_cluster_centroids[0], pca_cluster_centroids[1] 

    def get_cluster_labels(self):
        """
        Get the cluster labels assigned to each sample.
        
        Returns:
        - cluster_labels: The cluster label for each sample.
        """
        return self.clusters
    
    def get_cluster_dict_labels(self):
        """
        Get the cluster labels assigned to each sample.
        
        Returns:
        - dict_cluster_labels: The point-cluster pair for each sample.
        """
        dict_out = {}

        for i, point in enumerate(self.data.index):
            dict_out[point] = self.clusters[i]
    
        return dict_out


    def get_pca_components(self):
        """
        Get the PCA components used for dimensionality reduction.
        
        Returns:
        - pca_components: The PCA components (Principal Component 1 and 2).
        """
        return self.reduced_data
