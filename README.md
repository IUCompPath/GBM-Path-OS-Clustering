# Prognostic Stratification of GBM Patients by Unsupervised Clustering of Morphology Patterns on WSI

This repository includes the open-source code and associated documentation, of the approach described in the following manuscript. If you use this code or refer to this code, please include the appropriate citation.

## Training Steps

Follow the following steps to create patches, extract features, train the classifier to stratify the patients OS into low and high-risk.

```bash
DATA_DIRECTORY/
	├── slide_1.svs
	├── slide_2.svs
	└── ...
```

* `--custom_downsample`:
### Create Patches

``` shell
python create_patches_fp.py 
```

```bash
RESULTS_DIRECTORY/
	├── masks
    		├── slide_1.png
    		├── slide_2.png
    		└── ...
	├── patches
    		├── slide_1.h5
    		├── slide_2.h5
    		└── ...
	├── stitches
    		├── slide_1.png
    		├── slide_2.png
    		└── ...
	└── process_list_autogen.csv
```

### Extract Features

Code for features

### Train the Classifier

Code for training the classifier

### Calculation of Rand Index and Silhouette Scores

Additional code for Rand Index and Silhouette Scores.

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
