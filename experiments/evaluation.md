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

After downloading data following the instructions from the main README, you will see lerf_data_experiments, openscene_data, scannet and scanrefer_label.

| Folder                  | Description |
|-----------------------|-------------|
| lerf_data_experiments | Contains a config yaml that contains pointers to varies data path             |
| openscene_data        | Contains pre-processed Openscene features for each ScanNet scene            |
| scannet               | Contains source data for ScanNet scenes plus preprocessed LERF features     |
| scanrefer_label       | Contains the bbox labels for ScanRefer queries            |

Before running the jupyter notebooks, update the `root_directory` variable with the path to the scanrefer_label folder. We recommend running all of the jupyter notebook in the docker environment where all of the environment have been installed.

## Coordinate System Conversion

There are 3 coordinate systems: 
- A. OpenScene coordinate system
- B. LERF coordinate system
- C. Original Scannet coordinate system (your own custom dataset coordinate system)

### Evaluate on OpenScene:

To convert from coordinate system A to coordinate system C, coordinate system A has the same set of points in the same ording as in coordinate system C. To evaluate on ScanRefer using OpenScene, we can first look at which set of points is lit up in system A and those would be the set lit up in system C. And we can calculate mIOU in system C.

See this line for where we index into system C using the cluster we obtained from system A:
```
https://github.com/sled-group/chat-with-nerf/blob/b769c68e3862d8ea83f2756836f244d93ce4b980/chat_with_nerf/visual_grounder/picture_taker.py#L913
```

### Evaluate on LERF:

To convert from coordinate system B to coordinate system C, we need a transformation matrix which has been pre-calculated and provide in the data. And we can calculate mIOU in system C.

We use the transformation matrix to transform bbox in system B into system C:
```
https://github.com/sled-group/chat-with-nerf/blob/b769c68e3862d8ea83f2756836f244d93ce4b980/chat_with_nerf/visual_grounder/picture_taker.py#L173
```
As for how to obtain the transformation matrix, please refer to [Preprocess section](https://github.com/sled-group/chat-with-nerf?tab=readme-ov-file#preprocess--preprare-your-own-data).
