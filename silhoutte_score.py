# -*- coding: utf-8 -*-
"""
Created on Sun Oct 4 10:37:51 2022

Author: Shubham
"""

import os
from glob import glob
import numpy as np
import pickle
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import pathlib

# Configuration

ROOT_DIR = 'path/to/root_dir'
FOLDER_NAMES = ['0', '1']
OUTPUT_DIR = '/path/to/output/dir'

# Function to compute features
def compute_features(root_dir, folder_names):
    train_features = []
    patch_list = []
    
    print('Computing Features...')
    
    for folder_name in folder_names:
        class_folders = glob(os.path.join(root_dir, folder_name, '*'))
        for folder in class_folders:
            feature_path = os.path.join(folder, f'{os.path.basename(folder)}_VGG16_256.npy')
            feature_mat = np.load(feature_path)
            feature_mat = np.squeeze(feature_mat)
            train_features.append(feature_mat)

            patch_file = os.path.join(folder, f'{os.path.basename(folder)}_VGG16_256_patches_path.pkl')
            with open(patch_file, 'rb') as f:
                patches = pickle.load(f)
                patch_list.extend(patches)
    
    print('Computing Features done!')
    return np.vstack(train_features), patch_list

# Function for dimensionality reduction and clustering
def dimensionality_reduction_and_clustering(features, n_clusters):
    print('Dimensionality Reduction...')
    
    scaler = StandardScaler()
    scaler.fit(features)
    scaled_features = scaler.transform(features)

    pca = PCA(n_components=0.95, random_state=42)
    pca.fit(scaled_features)
    reduced_features = pca.transform(scaled_features)
    
    print(f'Number of components: {pca.n_components_}')
    
    print('Clustering...')
    kmeans = KMeans(n_clusters=n_clusters, n_init=100, random_state=42)
    labels = kmeans.fit_predict(reduced_features)
    
    silhouette_avg = silhouette_score(reduced_features, labels, random_state=1)
    print(f'Silhouette Score: {silhouette_avg}')
    
    return reduced_features, labels

# Function to save silhouette scores
def save_silhouette_scores(labels, reduced_features, patch_list, output_dir, n_clusters):
    print('Saving Silhouette Scores...')
    
    silhouette_scores = silhouette_samples(reduced_features, labels)
    
    output_dir = os.path.join(output_dir, f'silhouette_score_{n_clusters}_clusters')
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_patches = [patch_list[i] for i in cluster_indices]
        
        output_file = os.path.join(output_dir, f'{cluster_id}.pkl')
        with open(output_file, 'wb') as f:
            pickle.dump(cluster_patches, f)
    
    print('Silhouette Scores saved!')

# Main processing
def main():
    train_features, patch_list = compute_features(ROOT_DIR, FOLDER_NAMES)
    
    print(f'Number of samples: {len(train_features)}')
    print(f'Mean of features: {np.mean(train_features)}')
    
    reduced_features, labels = dimensionality_reduction_and_clustering(train_features, n_clusters=7)
    
    save_silhouette_scores(labels, reduced_features, patch_list, OUTPUT_DIR, n_clusters=7)

if __name__ == '__main__':
    main()
