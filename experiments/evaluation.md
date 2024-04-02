# Evaluations

After you download, you will see lerf_data_experiments, openscene_data, scannet and scanrefer_label.
| Folder                  | Description |
|-----------------------|-------------|
| lerf_data_experiments | Contains a config yaml that contains pointers to varies data path             |
| openscene_data        | Contains pre-processed Openscene features for each ScanNet scene            |
| scannet               | Contains source data for ScanNet scenes plus preprocessed LERF features     |
| scanrefer_label       | Contains the bbox labels for ScanRefer queries            |

Note to put scanrefer_label path for **root_directory** in each experiments.

## No GPT + LeRF

There is one function called act_no_gpt. Disable all the singleton code. Connect no gpt function with that.

One example:
```
chat-with-nerf/experiments/evaluation_without_gpt_lerf_final.ipynb
```

## With GPT + LeRF

Current code can handle but better disable singleton code.

One example:
```
chat-with-nerf/experiments/evaluate_lerf_14_with_gpt.ipynb
```

## No GPT + OpenScene

There is one function called act_no_gpt. Disable all the singleton code. Connect no gpt function with that.

One example:
```
chat-with-nerf/experiments/evaluate_baseline_openscene.ipynb
```

## With GPT + OpenScene

Current code can handle but better disable singleton code.

One example:
```
chat-with-nerf/experiments/evaluation_with_gpt_openscene_no_visual_feedback.ipynb
``` 
