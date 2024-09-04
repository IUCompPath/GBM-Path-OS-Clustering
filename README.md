# Prognostic Stratification of GBM Patients by Unsupervised Clustering of Morphology Patterns on WSI

This repository includes the open-source code and associated documentation, of the approach described in the following manuscript. If you use this code or refer to this code, please include the appropriate citation.

Use the environment configuration file to create a conda environment:
```shell
conda env create -f environment.yaml
```

Activate the environment:
```shell
conda activate clustering
```


## Training Steps

We collect all the slides and place in the DATA_DIRECTORY

```bash
DATA_DIRECTORY/
	├── slide_1.svs
	├── slide_2.svs
	└── ...
```

### Create Patches

We extract patches from WSI with GaNDLF. Details on how to perform patch extraction for histology images in the [GaNDLF documentation](https://docs.mlcommons.org/GaNDLF/usage/#offline-patch-extraction-for-histology-images-only).

Alternatively, you can follow the [CLAM](https://github.com/mahmoodlab/CLAM) patch extraction to store the coordinates of patches from WSI.

```bash
PATCH_DIRECTORY/
	├── slide_1.h5
	├── slide_1.h5
	├── slide_2.h5
	├── slide_1.h5
	└── ...
```

### Extract Features

Modify the code below according to patch extraction codebase. 

We store the coordinates in the h5 format with corresponding metadata.

We extract features using VGG16 pretrained on ImageNet to compress the WSI into feature vector. This are stored in the `.npy` format. We also store the corresponding patches index in `.pkl` file to retain patches information.



```shell
python extract_features.py --slide_dir /path/to/slides --csv_path /path/to/csv --root_dir /path/to/root_dir --output_dir /path/to/save/features/
```

```bash
FEATURE_DIRECTORY/
	├── class_0
		├── slide_1.npy
		├── slide_1_patches_path.pkl
		└── ...
	├── class_1
		├── slide_1.npy
		├── slide_1_patches_path.pkl
		└── ...
```

### Train the Classifier

Make sure to update the paths in the `ROOT_DIR`, `TRAIN_DIR`, `VAL_DIR`, `TEST_DIR`, and `RESULTS_DIR` variables to your specific directories.

```shell
python main.py
```

This will train various ML classifiers and store the evaluation metrices in csv in `RESULTS_DIR`



### Calculation of Rand Index and Silhouette Scores

Additional code for Rand Index:

```shell
python rand_index.py
```

For silhouette scores:

```shell
python silhoutte_score.py
```

## Citation

```
@article{Baheti_2024_gbm_path_os_clustering,
	author={Baheti, Bhakti and Innani, Shubham and Nasrallah, MacLean and Bakas, Spyridon},
	title={Prognostic Stratification of Glioblastoma Patients by Unsupervised Clustering of Morphology Patterns on Whole Slide Images Furthering Our Disease Understanding},
	journal={In Review},
	url={TBA},
	year={TBA},
	doi={TBA},
	publisher={TBA}
}
```

## Funding

Research reported in this publication was partly supported by the Informatics Technology for Cancer Research (ITCR) program of the National Cancer Institute (NCI) of the National Institutes of Health (NIH), under award number U01CA242871. The content of the publication is solely the responsibility of the authors and does not represent the official views of the NIH.
