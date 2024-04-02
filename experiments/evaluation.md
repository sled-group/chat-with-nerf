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
