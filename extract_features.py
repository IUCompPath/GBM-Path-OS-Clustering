# -*- coding: utf-8 -*-

"""
Created on Sun Oct 4 10:37:51 2022

@author: shubham
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import pathlib
import openslide
import cv2
from tqdm import tqdm
from glob import glob
from skimage import io
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
import pickle
from skimage.transform import rescale, resize
from skimage.color import rgb2hed
from skimage.exposure import rescale_intensity
import h5py
import os
from sklearn.cluster import KMeans

def get_wsi_features_all_patches(patient_id, n, model, attr_dict, slide_dir):
    """
    Extracts features from all patches of a given WSI using the specified model.

    Parameters:
    - patient_id: str, ID of the patient (corresponding to the slide file name).
    - n: int, number of patches to process.
    - model: Keras Model, pre-trained model for feature extraction.
    - attr_dict: dict, attributes dictionary containing patch size and other info.
    - slide_dir: str, directory where the slide images are located.

    Returns:
    - combine_features_np: numpy array, extracted features.
    - patches_list: list of str, list of patch identifiers.
    """
    combine_features = []
    patches_list = []

    for i in tqdm(range(n)):
        slide = openslide.OpenSlide(os.path.join(slide_dir, f'{patient_id}.svs'))
        patch_size = attr_dict['patch_size']
        x, y = ds_arr[i]
        patch_array = slide.read_region((x, y), 0, (patch_size, patch_size)).convert('RGB')
        patches_list.append(f'{patient_id}_{x}_{y}')
        if patch_size == 512:
            patch_array = patch_array.resize((256, 256))
        patch_array = np.array(patch_array)
        ihc_hed = rgb2hed(patch_array)
        patch_hsv_1 = cv2.cvtColor(patch_array, cv2.COLOR_RGB2HSV)[:, :, 0]
        e = rescale_intensity(ihc_hed[:, :, 1], out_range=(0, 255), in_range=(0, np.percentile(ihc_hed[:, :, 1], 99)))
        if len(e[e < 100]) / (256 * 256) > 0.9 or len(np.where(patch_hsv_1 < 128)[0]) / (256 * 256) > 0.95:
            continue
        img1 = img_to_array(patch_array)
        image1 = np.expand_dims(img1, axis=0)
        x = preprocess_input(image1)
        temp = model.predict(x)
        pooled_featuremap = np.squeeze(temp, axis=0)
        combine_features.append(pooled_featuremap)
        slide.close()

    print(f'Number of features extracted: {len(combine_features)}')
    combine_features_np = np.array(combine_features)
    return combine_features_np, patches_list


def cluster_cnn_features(combine_features, n_clusters, n):
    """
    Clusters CNN features using K-Means and calculates final feature vectors.

    Parameters:
    - combine_features: numpy array, extracted features from CNN.
    - n_clusters: int, number of clusters for K-Means.
    - n: int, number of samples.

    Returns:
    - final_feature_flatten: numpy array, flattened final feature vector.
    """
    

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(combine_features)
    temp_labels = kmeans.labels_
    templ_centroids = kmeans.cluster_centers_
    unique_values, occur_count = np.unique(temp_labels, return_counts=True)
    cluster_wt = occur_count / n
    final_feature = np.multiply(templ_centroids, cluster_wt[:, None])
    final_feature_flatten = final_feature.flatten()
    print(f'Sum of final features: {np.sum(final_feature_flatten)}')
    return final_feature_flatten


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Extract features from WSI patches.')
    parser.add_argument('--slide_dir', type=str, required=True, help='Directory where the slide images are located.')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the CSV file containing slide IDs.')
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory for saving the extracted features.')
    parser.add_argument('--output_dir', type=str, required=True, help='Root directory for saving the extracted features.')
    args = parser.parse_args()

    image_id = pd.read_csv(args.csv_path)['slide_id'].to_list()
    base_model = VGG16(include_top=False, weights="imagenet", input_shape=(256, 256, 3), pooling='avg')
    base_model.trainable = False
    model = tf.keras.Model(inputs=base_model.input, outputs=base_model.get_layer('global_average_pooling2d').output)

    for patient_id in tqdm(image_id):
        foldername = os.path.join(args.slide_dir, patient_id)
        folder_name = os.path.basename(foldername)
        print(folder_name)
        attr_dict = {}
        with h5py.File(f'{foldername}.h5', "r") as f:
            a_group_key = list(f.keys())[0]
            ds_arr = f[a_group_key][()]
            for k, v in f[a_group_key].attrs.items():
                attr_dict[k] = v

        total_patches = len(ds_arr)
        print(f'Total patches: {total_patches}')

        wsi_featuremap, patches_list = get_wsi_features_all_patches(folder_name, total_patches, model, attr_dict, args.slide_dir)
        
        output_dir = os.path.join(args.root_dir, args.output_dir, folder_name)
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

        np.save(os.path.join(output_dir, f'{folder_name}_VGG16_256.npy'), wsi_featuremap, allow_pickle=True)
        with open(os.path.join(output_dir, f'{folder_name}_VGG16_256_patches_path.pkl'), 'wb') as f:
            pickle.dump(patches_list, f)
