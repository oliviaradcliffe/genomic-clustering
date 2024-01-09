# Gene Expression Profile Clustering
# November 10, 2023
# by Olivia Radcliiffe

import os
import sys
import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt

# Applys KMeans clustering on data with number of clusters=n
#  using the sklearn package
def KMeans_implementation(data, n):
    from sklearn.cluster import KMeans

    # KMeans
    kmeans = KMeans(n_clusters=n, random_state=0, n_init="auto")
    y_pred = kmeans.fit_predict(data)

    return y_pred

# Applys Agglomerative Hierarchical Clustering on data with min (ward)
#  approach using the sklearn package
def AggloHier_implementation(data, n):
    from sklearn.cluster import AgglomerativeClustering

    # AgglomerativeClustering with min (ward) approach
    agglo_Hier = AgglomerativeClustering(n_clusters=n, linkage='ward')
    y_pred = agglo_Hier.fit_predict(data)

    return y_pred

# Computes the external indicies Adjusted Rand index) 
# and internal index (Silhoette Score) 
def calc_internal_external(data_X, data_ground_truth, data_y_pred):
    """
    Parameters: 
        data_X - Attributes of your data (gene expression values)
        data_ground_truth - Ground truth clustering labels
        data_y_pred - Predicted clustering labels
    Output:
        Prints the resulting Adjusted Rand index, and 
        Silhoette Score
    """

    # Caluclate Adjusted Rand index
    ad_rand_i = adjusted_rand_score(data_ground_truth, data_y_pred)
    print("Adjusted Rand Index: ", round(ad_rand_i, 2))

    # Calculate Silhouette score
    sil = silhouette_score(data_X, data_y_pred, metric='euclidean')
    print("Silhouette Score: ",round(sil, 2))


# Determines 3 PCA components for 3d plotting using sklearn PCA
def PCA_components(data):
    from sklearn.decomposition import PCA

    # 3 PCA components for 3d plotting
    pca = PCA(n_components=3)
    threeDpca = pca.fit_transform(data)

    return threeDpca

# Plots data in axs subplot (does not show figure)
def plot_results(data, axs, labels, title):
    """
    Parameters: 
        data   - Attributes of your data (gene expression values)
        axs    - Subplot you want to plot on
        labels - Clustering labels (used for coloring)
        title  - Title of plot
    Output:
        Adds new plot to axs
    """

    # extract the first three PCA components
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]

    # create the scatter plot
    axs.scatter(x, y, z, marker='o', c=labels)

    # set labels and title
    axs.set_xlabel('PCA Component 1')
    axs.set_ylabel('PCA Component 2')
    axs.set_zlabel('PCA Component 3')
    axs.set_title(title)


# Plots datasets and clustering results with Principal Component Analysis (PCA). 
def plot_clustering(fig, plotIndicies, data_X, data_ground_truth, datasetName, clustering_func1, clustering_func2):
    """
    Parameters:
        fig - matplotlib figure for plots
        plotIndicies - list of subplot indicies
        data_X - Attributes of your data (gene expression values)
        data_ground_truth - Ground truth clustering labels
        datasetName - Name of dataset (for plot title)
        clustering_func1 - First clustering implementation function
        clustering_func2 - Second clustering implementation function
    Output:
        Adds 3 subplots (dataset, clustering 1 result, clustering 2 result) to figure
    """

    # Reduce data with PCA
    reduced_data = PCA_components(data_X)
    # plot ground truth labels
    ax = fig.add_subplot(2, 3, plotIndicies[0], projection='3d')
    plot_results(reduced_data, ax, data_ground_truth, title=(datasetName +" Ground Truth"))

    # Apply clustering methods
    numClusters = len(np.unique(data_ground_truth))
    y_pred1 = clustering_func1(reduced_data, numClusters)
    y_pred2 = clustering_func2(reduced_data, numClusters)

    # plot ground truth and cluster predictions for data
    ax2 = fig.add_subplot(2, 3, plotIndicies[1], projection='3d')
    plot_results(reduced_data, ax2,  y_pred1, title=clustering_func1.__name__)
    ax3 = fig.add_subplot(2, 3, plotIndicies[2], projection='3d')
    plot_results(reduced_data, ax3,  y_pred2, title=clustering_func2.__name__)

# Validates clustering methods by computing internal and external index metrics and
# visualizes the datasets and clustering results with PCA
def validate_clustering(dataset1, data1Name, dataset2, data2Name, data1_ground_truth, data2_ground_truth, clustering_func1, clustering_func2):
    """
    Parameters:
        dataset1 - Attributes of first (Cho) dataset
        data1Name - Name of dataset1
        dataset2 - Attributes of second (iyer) dataset
        data2Name - Name of dataset2
        data1_ground_truth - Ground truth cluster labels of data1 dataset
        data2_ground_truth - Ground truth cluster labels of data2 dataset
        clustering_func1 - First clustering implementation function
        clustering_func2 - Second clustering implementation function
    Output:
        Figure with 6 3d PCA plots showing the 2 datasets and the clustering
          results from 2 clustering methods
    """

    # Apply clustering1 to cho and iyer data
    cho_y_pred_1 = clustering_func1(dataset1, len(np.unique(data1_ground_truth)))
    iyer_y_pred_1 = clustering_func1(dataset2, len(np.unique(data2_ground_truth)))

    # Apply clustering2 to cho and iyer data
    cho_y_pred_2 = clustering_func2(dataset1, len(np.unique(data1_ground_truth)))
    iyer_y_pred_2 = clustering_func2(dataset2, len(np.unique(data2_ground_truth)))


    # VALIDATE CLUSTERING 1 - Kmeans
    print("\n------------" + clustering_func1.__name__ + "------------")
    # Cho dataset
    print("*******************" + data1Name + "****************************")
    calc_internal_external(dataset1, data1_ground_truth, cho_y_pred_1)
    print("*******************" + data2Name + "***************************")
    # Iyer dataset
    calc_internal_external(dataset2, data2_ground_truth, iyer_y_pred_1)

    # VALIDATE CLUSTERING 2 - Hierarchical
    print("\n------------" + clustering_func2.__name__ + "------------")
    # Cho dataset
    print("*******************" + data1Name + "****************************")
    calc_internal_external(dataset1, data1_ground_truth, cho_y_pred_2)
    print("*******************" + data2Name + "***************************")
    # Iyer dataset
    calc_internal_external(dataset2, data2_ground_truth, iyer_y_pred_2)


    # VISUALIZE DATA AND CLUSTERING RESULTS

    # Create 3D scatter plots
    fig = plt.figure(figsize=(12, 8))
    plot_clustering(fig, [1,2,3], dataset1, data1_ground_truth, data1Name, clustering_func1, clustering_func2)
    plot_clustering(fig, [4,5,6], dataset2, data2_ground_truth, data2Name, clustering_func1, clustering_func2)
    
    plt.show()




def main():

    # PROCESS DATA

    # import two gene datasets (cho and iyer) for this assignment
    directory = os.path.dirname(os.path.abspath(sys.argv[0]))
    cho_data = pd.read_csv(directory + "/cho.txt", sep='\t', header=None)
    iyer_data = pd.read_csv(directory + "/iyer.txt", sep='\t', header=None)

    # remove outliers
    cho_data_processed = cho_data.drop(cho_data.loc[cho_data[1]==-1].index)
    iyer_data_processed = iyer_data.drop(iyer_data.loc[iyer_data[1]==-1].index)

    # Extract gene expression values (attributes)
    cho_X = cho_data_processed.iloc[:, 2:]
    iyer_X = iyer_data_processed.iloc[:, 2:]

    # standardize instance
    scaler = StandardScaler()
    # standardize data
    cho_X = scaler.fit_transform(cho_X.iloc[:,2:])
    iyer_X = scaler.fit_transform(iyer_X.iloc[:,2:])

    # extract ground truth cluster labels
    cho_ground_truth = np.array(cho_data_processed.iloc[:, 1])
    iyer_ground_truth = np.array(iyer_data_processed.iloc[:, 1])


    # VALIDATE CLUSTERING

    validate_clustering(cho_X, "Cho", iyer_X, "Iyer", cho_ground_truth, iyer_ground_truth, KMeans_implementation, AggloHier_implementation)
    


if __name__ == "__main__":
    main()