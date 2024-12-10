import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

from Utils import scaler_data_standard
from Utils import inverse_scaler_data_standard
from Utils import compute_distance_from_centroids


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
            (self.data_scaled, self.scaler)  = scaler_data_standard(data)
        else:
            self.data_scaled = data

        # Perform clustering
        self.clustering_model = AgglomerativeClustering(n_clusters=self.n_clusters, linkage=self.linkage_method)
        self.clusters_labels = self.clustering_model.fit_predict(self.data_scaled)
        
        # Perform PCA reduction on scaled input data
        self.reduced_data = self._perform_PCA(self.data_scaled)[0]

        # Perform PCA reduction on scaled input data and centroids
        self.df_data_centroids_scaled = self._merge_scaled_data_centroids()
        self.reduced_data_centroids = self._perform_PCA(self.df_data_centroids_scaled)[0]


    def _merge_scaled_data_centroids(self):
        # Create dataframe with scaled data
        df_scaled_data = pd.DataFrame(self.data_scaled, columns=list(self.data.columns), index=self.data.index)
        
        # Retrieve df centroids
        df_centroids = self.get_cluster_centroids()

        # Merge the two dataframes
        df_merged = pd.concat([df_scaled_data, df_centroids])
        
        return df_merged

    def _perform_PCA(self, data):
        # If the input is a numpy array, assign default names like 'Feature_1', 'Feature_2', ...
        feature_names = list(self.data.columns)
    
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
        

    def plot_scatter(self, size:tuple=(10, 7)):
        """
        Plot the clusters in the 2D PCA-reduced space.
        """
        # Obtain clusters and centroids numbers
        
        clusters_and_centroids = np.append(self.clusters_labels, list(range(self.n_clusters)))

        # Plot the graph
        plt.figure(figsize=size)
        plt.scatter(self.reduced_data_centroids[:, 0], self.reduced_data_centroids[:, 1], c=clusters_and_centroids, cmap='viridis') 
        
        # Add labels for each data point
        for i, label in enumerate(self.df_data_centroids_scaled.index):  # fix enumerate
            plt.text(self.reduced_data_centroids[i, 0], 
                     self.reduced_data_centroids[i, 1], 
                     label, fontsize=10, ha='right', va='bottom')
        # Set equal scaling for both axes
        plt.gca().set_aspect('equal', adjustable='box')
    
        plt.title(f"Agglomerative Clustering (n_clusters={self.n_clusters}) - 2D PCA")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.colorbar(label='Cluster')
        plt.show()

    def get_cluster_centroids(self):
        """
        Get the centroids of the clusters by calculating the mean of each feature for each cluster.
        
        Returns:
        - PCA-reduced centroids: An array of the PCA components of the centroids.
        """
        index_centroids = [f"Centroid_{i}" for i in list(range(self.n_clusters))]
        cluster_centroids = pd.DataFrame(columns=list(self.data.columns), index=index_centroids)
        
        for i in range(self.n_clusters):
            cluster_data = self.data_scaled[self.clusters_labels == i]
            centroid = np.mean(cluster_data, axis=0)
            cluster_centroids.loc[index_centroids[i]] = centroid

        return cluster_centroids

    def get_cluster_labels(self):
        """
        Get the cluster labels assigned to each sample.
        
        Returns:
        - cluster_labels: The cluster label for each sample.
        """
        return self.clusters_labels
    
    def get_pca_components(self):
        """
        Get the PCA components used for dimensionality reduction.
        
        Returns:
        - pca_components: The PCA components (Principal Component 1 and 2).
        """
        return self.reduced_data
    
    def get_descaled_dataframe(self)-> pd.DataFrame:
        """
        Revert the scaled data back to its original scale.

        This function uses the fitted scaler to apply an inverse transformation 
        to the scaled data (`df_data_centroids_scaled`) and return it in its 
        original form.

        Returns:
        - pd.DataFrame: A DataFrame with the original (unscaled) values, having the 
        same shape, index, and columns as `df_data_centroids_scaled`.
        """
        df_out = inverse_scaler_data_standard(self.df_data_centroids_scaled, self.scaler)
        return df_out

    def compute_wcd(self) -> float:
        """
        Calculate the Within-Cluster Dispersion (WCD).

        Returns:
        - float: The WCD value, representing the average dispersion within clusters.
        """
        # Assign attributes to variables for better handling
        data = self.data_scaled
        centroids = self.get_cluster_centroids().values
        labels = self.clusters_labels

        # Initialize variables to store results
        total_dispersion = 0.0
        total_points = 0

        # Loop over centroids and compute distances
        for i, centroid in enumerate(centroids):
            # Convert to array
            centroid = np.asarray(centroid, dtype=np.float64)

            cluster_points = data[labels == i]
            cluster_points = np.asanyarray(cluster_points, dtype=np.float64)

            distances = np.linalg.norm(cluster_points - centroid, axis=1)

            total_dispersion += np.sum(distances)

            total_points += len(cluster_points)

        return total_dispersion / total_points if total_points > 0 else 0.0

    def compute_centroids_spread(self):   ### To do!

        # Compute the average distance 
        dist_segments_brands = compute_distance_from_centroids(df_observations=df_segments, df_centroids=df_brands)
        res = dist_segments_brands[3]
        
        res_dict = {}

        # iterate over keys
        for key in res.keys():
            mean_dist = np.mean(list(res[key].values()))
            res_dict[key] = mean_dist

        return res_dict