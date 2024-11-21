import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

def labels_optimal_mapping(cluster_labels, true_labels):
    row_ind, col_ind = optimal_label_mapping(cluster_labels, true_labels)
    mapping = dict(zip(row_ind, col_ind))
    updated_cluster_labels = update_cluster_labels(cluster_labels, mapping)
    return updated_cluster_labels

def calculate_cost_matrix(clusters, true_labels):
    """
    Calculate the cost matrix for the Hungarian algorithm.
    Cost is the number of mismatched elements for each cluster-label pair.
    """
    max_label = max(np.max(clusters), np.max(true_labels)) + 1
    cost_matrix = np.zeros((max_label, max_label))

    for i in range(max_label):
        for j in range(max_label):
            cost_matrix[i, j] = np.sum((clusters == i) != (true_labels == j))

    return cost_matrix

def optimal_label_mapping(clusters, true_labels):
    """
    Find the optimal mapping of cluster labels to true labels.
    """
    cost_matrix = calculate_cost_matrix(clusters, true_labels)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return row_ind, col_ind

def update_cluster_labels(cluster_labels, mapping):
    updated_labels = np.copy(cluster_labels)
    for cluster, true_label in mapping.items():
        updated_labels[cluster_labels == cluster] = true_label
    return updated_labels

