# Evaluations

There will be in total 4 sets of experiments and we organize each experiment in one jupyter notebook. Please use the non-interactive prompt for evaluation.

### No GPT + LeRF

```
chat-with-nerf/experiments/evaluation_without_gpt_lerf_final.ipynb
```

### With GPT + LeRF

```
chat-with-nerf/experiments/evaluate_lerf_14_with_gpt.ipynb
```

### No GPT + OpenScene

```
chat-with-nerf/experiments/evaluate_baseline_openscene.ipynb
```

### With GPT + OpenScene

```
chat-with-nerf/experiments/evaluation_with_gpt_openscene_no_visual_feedback.ipynb
```

## Data Preparation

After downloading data following the instructions from the main README, you will see lerf_data_experiments, openscene_data, ScanNet and scanrefer_label.

| Folder                  | Description |
|-----------------------|-------------|
| lerf_data_experiments | Contains a config yaml that contains pointers to varies data path             |
| openscene_data        | Contains pre-processed Openscene features for each ScanNet scene            |
| scannet               | Contains source data for ScanNet scenes plus preprocessed LERF features     |
| scanrefer_label       | Contains the bbox labels for ScanRefer queries            |

Before running the Jupyter notebooks, update the `root_directory` variable with the path to the scanrefer_label folder. We recommend running all of the Jupyter notebooks in the docker environment where all of the environments have been installed.

## Coordinate System Conversion

There are 3 coordinate systems: 
- A. OpenScene coordinate system
- B. LERF coordinate system
- C. Original ScanNet coordinate system (or your custom dataset coordinate system)

### Evaluate on OpenScene:

To convert from coordinate system A to coordinate system C, coordinate system A has the same set of points in the same ordering as in coordinate system C. To evaluate ScanRefer using OpenScene, we can first look at the points grounded by OpenScene in system A, and use those points' indices to index into system C. Then, we can calculate mIoU in system C.

See [this line](https://github.com/sled-group/chat-with-nerf/blob/b769c68e3862d8ea83f2756836f244d93ce4b980/chat_with_nerf/visual_grounder/picture_taker.py#L913) for where we index into system C using the cluster we obtained from system A:

### Evaluate on LERF:

To convert from coordinate system B to coordinate system C, we need a transformation matrix that has been pre-calculated and provided in the downloadable pre-processed data. Then, we can calculate mIoU in system C.

See [this line](https://github.com/sled-group/chat-with-nerf/blob/b769c68e3862d8ea83f2756836f244d93ce4b980/chat_with_nerf/visual_grounder/picture_taker.py#L173) for where we use the transformation matrix to transform b-boxes in system B into system C:

As for how to obtain the transformation matrix, please refer to [Preprocess section](https://github.com/sled-group/chat-with-nerf?tab=readme-ov-file#preprocess--preprare-your-own-data).
