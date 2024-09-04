# -*- coding: utf-8 -*-
"""
Created on Sun Oct 4 10:37:51 2022

Author: Shubham
"""

import os
from glob import glob
import numpy as np
import pickle
from sklearn.metrics import silhouette_samples, silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from itertools import permutations

# Configuration
ROOT_DIR = '/path/to/root/dir/'
FOLDER_NAMES = ['0', '1']
LABELS_DIR = os.path.join('clustering/rand_index/labels/7_clusters')
RESULTS_DIR = os.path.join('clustering/results_10')

# Compute features
def compute_features(root_dir, folder_names):
    train_features = []
    patch_list = []

    print('Computing Features...')
    for folder_name in folder_names:
        class_folders = os.listdir(os.path.join(root_dir, folder_name))
        for folder in class_folders:
            feature_path = os.path.join(root_dir, folder_name, folder, f'{folder}_VGG16_256.npy')
            feature_mat = np.load(feature_path)
            feature_mat = np.squeeze(feature_mat)
            train_features.append(feature_mat)

            patch_file = os.path.join(root_dir, folder_name, folder, f'{folder}_VGG16_256_patches_path.pkl')
            with open(patch_file, 'rb') as f:
                patches = pickle.load(f)
                patch_list.extend(patches)
    
    print('Computing Features done!')
    return np.vstack(train_features), patch_list

# Dimensionality Reduction
def dimensionality_reduction(features, n_components=10):
    print('Dimensionality Reduction...')
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    pca = PCA(n_components=n_components, random_state=42)
    reduced_features = pca.fit_transform(scaled_features)
    
    print(f'Dimensionality Reduction is Done. Components: {n_components}')
    return reduced_features

# Clustering
def cluster_features(features, max_clusters=1000):
    global_labels = []

    for n_clusters in tqdm(range(1, max_clusters + 1)):
        kmeans = KMeans(n_clusters=n_clusters, n_init=100, random_state=np.random.randint(1, 5000))
        labels = kmeans.fit_predict(features)
        global_labels.append(labels)
    
    return global_labels

# Compute Adjusted Rand Index (ARI)
def compute_ari(narray):
    print('Computing Adjusted Rand Index...')
    
    val = list(range(len(narray)))
    perm_set = list(permutations(val, 2))
    average_rand_index = []

    def batch_works(i):
        array_1 = np.load(narray[perm_set[i][0]])
        array_2 = np.load(narray[perm_set[i][1]])
        return adjusted_rand_score(array_1, array_2)
    
    with Pool(processes=cpu_count() - 4) as pool:
        results = pool.map(batch_works, range(len(perm_set)))
    
    average_rand_index.extend(results)
    
    print(f'Average ARI: {np.average(average_rand_index)}')
    print(f'Min ARI: {min(average_rand_index)}')
    print(f'Max ARI: {max(average_rand_index)}')

# Process clustering results
def process_clustering_results(results_dir):
    all_df_paths = sorted(glob(os.path.join(results_dir, 'dt*.csv')))
    df_dt = pd.DataFrame(columns=[
        'Cluster', 'Training Acc', 'Val Acc', 'Test Acc', 'Train Sensitivity', 'Train Specificity',
        'Val Sensitivity', 'Val Specificity', 'Test Sensitivity', 'Test Specificity', 'Model'
    ])

    for csv_path in all_df_paths:
        split_df = pd.read_csv(csv_path)
        each_df = split_df.loc[split_df['Cluster'] == 5]
        df_dt = pd.concat([df_dt, each_df], ignore_index=True)
    
    df_dt.to_csv(os.path.join(results_dir, 'dt_cluster_5_combined.csv'), index=False)

# Main processing
def main():
    train_features, patch_list = compute_features(ROOT_DIR, FOLDER_NAMES)
    reduced_features = dimensionality_reduction(train_features)
    
    global_labels = cluster_features(reduced_features)
    
    narray = sorted(glob(os.path.join(LABELS_DIR, '*')))
    compute_ari(narray)
    
    process_clustering_results(RESULTS_DIR)

if __name__ == '__main__':
    main()
