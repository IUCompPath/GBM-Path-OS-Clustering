# -*- coding: utf-8 -*-
"""
Created on Sun Oct 4 10:37:51 2022

Author: Shubham
"""

import os
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

# Configuration
SPLIT = 9
ROOT_DIR = 'path_to_features_directory'
TRAIN_DIR = f'path_to_splits_directory/splits_{SPLIT}/train'
VAL_DIR = f'path_to_splits_directory/splits_{SPLIT}/val'
TEST_DIR = f'path_to_splits_directory/splits_{SPLIT}/test'
RESULTS_DIR = 'path_to_results_directory'

# Function to calculate Bag of Features (BoF)
def calculate_bof(root_dir, n_clusters, kmeans, scaler, pca_model):
    """
    Calculate Bag of Features (BoF) for each directory.
    
    Args:
        root_dir (str): Path to the directory containing the data.
        n_clusters (int): Number of clusters for KMeans.
        kmeans (KMeans): Fitted KMeans model.
        scaler (StandardScaler): Fitted scaler for feature normalization.
        pca_model (PCA): Fitted PCA model.
    
    Returns:
        tuple: Features and labels as numpy arrays.
    """
    folder_names = os.listdir(root_dir)
    bof_features = []
    bof_labels = []

    for class_name in folder_names:
        class_folders = os.listdir(os.path.join(root_dir, class_name))
        for folder_name in class_folders:
            feature_path = os.path.join(ROOT_DIR, folder_name, f'{folder_name}_VGG16_256.npy')
            feature_mat = np.squeeze(np.load(feature_path))
            scaled_data = scaler.transform(feature_mat)
            pca_transformed = pca_model.transform(scaled_data)

            # Predict cluster assignments
            cluster_assignments = kmeans.predict(pca_transformed)
            cluster_features = [np.mean(pca_transformed[cluster_assignments == c], axis=0) 
                                if np.any(cluster_assignments == c) else np.zeros(pca_transformed.shape[1]) 
                                for c in range(n_clusters)]
            
            # Create histogram of cluster counts
            bof_histogram = np.histogram(cluster_assignments, bins=list(range(n_clusters + 1)), density=True)[0]
            bof_features.append(bof_histogram.flatten())
            bof_labels.append(int(class_name))

    return np.array(bof_features), np.array(bof_labels)

# Main processing
def main():
    print('Computing Features...')
    
    # Load training features and labels
    train_features = []
    patch_list = []
    for class_name in ['0', '1']:
        class_folders = os.listdir(os.path.join(TRAIN_DIR, class_name))
        for folder_name in class_folders:
            feature_path = os.path.join(ROOT_DIR, folder_name, f'{folder_name}_VGG16_256.npy')
            feature_mat = np.squeeze(np.load(feature_path))
            train_features.append(feature_mat)
            
            patch_path = os.path.join(ROOT_DIR, folder_name, f'{folder_name}_VGG16_256_patches_path.pkl')
            with open(patch_path, 'rb') as f:
                patch_list.extend(pickle.load(f))

    train_features_all = np.vstack(train_features)
    
    # PCA for dimensionality reduction
    print('Dimensionality Reduction...')
    scaler = StandardScaler().fit(train_features_all)
    scaled_features = scaler.transform(train_features_all)
    
    pca = PCA(n_components=32, random_state=42)
    pca.fit(scaled_features)
    transformed_features = pca.transform(scaled_features)

    # Cluster evaluation
    range_n_clusters = list(range(2, 11))  # Adjust as needed
    df_dt = pd.DataFrame(columns=['Cluster', 'Training Acc', 'Val Acc', 'Test Acc', 
                                  'Train Sensitivity', 'Train Specificity', 
                                  'Val Sensitivity', 'Val Specificity', 
                                  'Test Sensitivity', 'Test Specificity', 'Model'])
    df_xgb = df_dt.copy()
    df_rf = df_dt.copy()
    
    for n_clusters in tqdm(range_n_clusters):
        print(f"Processing n_clusters: {n_clusters}")

        # KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, n_init=100, random_state=42)
        kmeans.fit(transformed_features)
        
        print('Computing Bag of Features...')
        X_train, Y_train = calculate_bof(TRAIN_DIR, n_clusters, kmeans, scaler, pca)
        X_val, Y_val = calculate_bof(VAL_DIR, n_clusters, kmeans, scaler, pca)
        X_test, Y_test = calculate_bof(TEST_DIR, n_clusters, kmeans, scaler, pca)
        print('Bag of Features done!')

        # Decision Tree Classifier
        print('Training Decision Tree...')
        dt_params = {'max_depth': [2, 3, 4, 5, 6, 7, 10],
                     'min_samples_leaf': [1, 2, 3, 5],
                     'min_samples_split': [2, 5, 7, 10],
                     'criterion': ["gini", "entropy"]}
        dt = DecisionTreeClassifier(random_state=42)
        dt_grid = GridSearchCV(dt, param_grid=dt_params, cv=10, verbose=2, n_jobs=-1, scoring='accuracy')
        dt_grid.fit(X_train, Y_train)
        best_dt = dt_grid.best_estimator_

        # Evaluate Decision Tree
        for clf, name in [(best_dt, 'Decision Tree')]:
            for X_data, Y_data, label in [(X_train, Y_train, 'Training'),
                                           (X_val, Y_val, 'Validation'),
                                           (X_test, Y_test, 'Testing')]:
                preds = clf.predict(X_data)
                acc = accuracy_score(Y_data, preds)
                tn, fp, fn, tp = confusion_matrix(Y_data, preds).ravel()
                sensitivity = tp / (tp + fn)
                specificity = tn / (tn + fp)
                
                print(f"{name} {label} accuracy: {acc:.4f}")
                
                if name == 'Decision Tree':
                    df_dt = df_dt.append({'Cluster': n_clusters, 'Training Acc': acc, 'Val Acc': acc,
                                          'Test Acc': acc, 'Train Sensitivity': sensitivity, 
                                          'Train Specificity': specificity, 'Val Sensitivity': sensitivity,
                                          'Val Specificity': specificity, 'Test Sensitivity': sensitivity,
                                          'Test Specificity': specificity, 'Model': clf}, ignore_index=True)
        
        # XGBoost Classifier
        print('Training XGBoost...')
        xgb_params = {'n_estimators': [5, 10, 15, 20, 50, 100],
                      'max_depth': [2, 3, 4, 5, 10, 25],
                      'gamma': [0, 1, 2],
                      'min_child_weight': [0, 1, 2, 3, 4, 5],
                      'subsample': [0.2, 0.3, 0.1, 0.15],
                      'colsample_bytree': [0.5, 0.3, 0.2, 0.15]}
        xgb = XGBClassifier(objective='binary:logistic', booster='gbtree', random_state=42)
        xgb_grid = GridSearchCV(xgb, param_grid=xgb_params, cv=10, verbose=2, n_jobs=-1, scoring='accuracy')
        xgb_grid.fit(X_train, Y_train)
        best_xgb = xgb_grid.best_estimator_

        # Evaluate XGBoost
        for clf, name in [(best_xgb, 'XGBoost')]:
            for X_data, Y_data, label in [(X_train, Y_train, 'Training'),
                                           (X_val, Y_val, 'Validation'),
                                           (X_test, Y_test, 'Testing')]:
                preds = clf.predict(X_data)
                acc = accuracy_score(Y_data, preds)
                tn, fp, fn, tp = confusion_matrix(Y_data, preds).ravel()
                sensitivity = tp / (tp + fn)
                specificity = tn / (tn + fp)
                
                print(f"{name} {label} accuracy: {acc:.4f}")

                if name == 'XGBoost':
                    df_xgb = df_xgb.append({'Cluster': n_clusters, 'Training Acc': acc, 'Val Acc': acc,
                                            'Test Acc': acc, 'Train Sensitivity': sensitivity, 
                                            'Train Specificity': specificity, 'Val Sensitivity': sensitivity,
                                            'Val Specificity': specificity, 'Test Sensitivity': sensitivity,
                                            'Test Specificity': specificity, 'Model': clf}, ignore_index=True)
        
        # Random Forest Classifier
        print('Training Random Forest...')
        rf_params = {'bootstrap': [True, False],
                     'max_depth': [2, 4, 6, 3, 5, 8, 9, 10, 20],
                     'max_features': [2, 3, 4, 5, 7, 8, 9, 10],
                     'min_samples_leaf': [1, 2, 3, 4],
                     'min_samples_split': [2, 5, 10],
                     'n_estimators': [5, 10, 15, 20, 50, 100]}
        rf = RandomForestClassifier(random_state=42)
        rf_grid = GridSearchCV(rf, param_grid=rf_params, cv=10, verbose=2, n_jobs=-1, scoring='accuracy')
        rf_grid.fit(X_train, Y_train)
        best_rf = rf_grid.best_estimator_

        # Evaluate Random Forest
        for clf, name in [(best_rf, 'Random Forest')]:
            for X_data, Y_data, label in [(X_train, Y_train, 'Training'),
                                           (X_val, Y_val, 'Validation'),
                                           (X_test, Y_test, 'Testing')]:
                preds = clf.predict(X_data)
                acc = accuracy_score(Y_data, preds)
                tn, fp, fn, tp = confusion_matrix(Y_data, preds).ravel()
                sensitivity = tp / (tp + fn)
                specificity = tn / (tn + fp)
                
                print(f"{name} {label} accuracy: {acc:.4f}")
                
                if name == 'Random Forest':
                    df_rf = df_rf.append({'Cluster': n_clusters, 'Training Acc': acc, 'Val Acc': acc,
                                          'Test Acc': acc, 'Train Sensitivity': sensitivity, 
                                          'Train Specificity': specificity, 'Val Sensitivity': sensitivity,
                                          'Val Specificity': specificity, 'Test Sensitivity': sensitivity,
                                          'Test Specificity': specificity, 'Model': clf}, ignore_index=True)

    # Save results
    df_dt.to_csv(os.path.join(RESULTS_DIR, f'dt_results_split_{SPLIT}.csv'), index=False)
    df_xgb.to_csv(os.path.join(RESULTS_DIR, f'xgb_results_split_{SPLIT}.csv'), index=False)
    df_rf.to_csv(os.path.join(RESULTS_DIR, f'rf_results_split_{SPLIT}.csv'), index=False)

if __name__ == '__main__':
    main()
