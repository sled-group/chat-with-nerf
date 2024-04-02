# :camera_flash: Chat with NeRF: Grounding 3D Objects in Neural Radiance Field through Dialog

[![Project](https://img.shields.io/badge/Project-Page-20B2AA.svg)](https://chat-with-nerf.github.io/)
[![Paper](https://img.shields.io/badge/arXiv-Paper-red.svg)](https://arxiv.org/abs/2309.12311)
[![Video](https://img.shields.io/badge/YouTube-Video-red)](https://youtu.be/eO-Vaf-1R1s)
[![Demo](https://img.shields.io/badge/Website-Demo-ff69b4.svg)](http://sled-whistler.eecs.umich.edu:7777/)
[![Embark](https://img.shields.io/badge/Discord-Chat--with--NeRF-%237289da.svg?logo=discord)](https://discord.gg/8rT4GpRq9g)


![Demo of Chat-with-NeRF](https://github.com/sled-group/chat-with-nerf/assets/12980222/6a327112-abbd-4b6a-ba22-e7e254a7fe6c "Overview")


## :bulb: Highlight

- **Open-Vocabulary 3D Localization.** Locate **anything** with natural language dialog!
- **Interactive Grounding.** Humans will be able to chat with an agent to localize novel objects.

## :fire: News
- [2023-09-21] We released a paper [LLM-Grounder: Open-Vocabulary 3D Visual Grounding with Large Language Model as an Agent
](https://arxiv.org/abs/2309.12311) that includes more quantitative evaluations of our pipeline!
- [2023-05-31] We improved the demo by adding grounding result visualization in 3D, taking pictures in real time, and speeding up inference by parallelization. Try out the [new demo](http://sled-whistler.eecs.umich.edu:7777/)!
- [2023-05-15] The first version of chat-with-nerf is available now! Please try out [demo](http://sled-whistler.eecs.umich.edu:7777/)!

## :label: TODO
- [x] A faster process to determine camera poses and rendering pictures. See discussion [#15](https://github.com/sled-group/chat-with-nerf/issues/15). Implemented in [#17](https://github.com/sled-group/chat-with-nerf/pull/17).
- [x] Use [LLaVA](https://llava-vl.github.io/) to replace BLIP-2 for better image captioning.
- [ ] Improve the foundation model (currently CLIP is used) used in LERF for grounding, which can potentially improve spatial and affordance understanding. Potential candidate: [LLaVA](https://llava-vl.github.io/), [BLIP-2](https://huggingface.co/docs/transformers/main/model_doc/blip-2), [OWL-ViT](https://huggingface.co/docs/transformers/model_doc/owlvit).

## :hammer_and_wrench: Install

To install the dependencies we provide a Dockerfile:
```bash
docker build -t chat-with-nerf:latest .
```
Or if you want to pull remote image from Dockerhub to save significant time, please try:
```bash
docker pull jedyang97/chat-with-nerf:latest
```

Otherwise, if you prefer build it locally:
```bash
conda create --name nerfstudio -y python=3.8
conda activate nerfstudio
pip install torch==1.13.1 torchvision functorch --extra-index-url https://download.pytorch.org/whl/cu117
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install nerfstudio

git clone https://github.com/kerrj/lerf
python -m pip install -e .
ns-train -h
```
Note that specific CUDA 11.3 is required. For further information, please check nerfstudio installation
guide.

Then locally you need to run
```bash
git clone https://github.com/sled-group/chat-with-nerf.git
```


Download and construct the [llava-13b-v0](https://huggingface.co/liuhaotian/LLaVA-13b-delta-v0) checkpoint (see [LLaVA's documentation on how to construct the checkpoint](https://github.com/haotian-liu/LLaVA/tree/8b21169def6c8ed57afa5e7bf790c1a91b530940#llava-13b)). Then assuming you store the constructed `llava-13b-v0` checkpoint under `<my_path_to_llava>/llava-13b-v0`, move the checkpoint to `/chat-with-nerf/pre-trained-weights/LLaVA`. 
```
cd chat-with-nerf
mkdir -p pre-trained-weights/LLaVA
cd pre-trained-weights/LLaVA
mv <my_path_to_llava>/llava-13b-v0 .
```

Alternatively, you can supply a different version of LLaVA checkpoint and change `LLAVA_PATH`'s value in `chat_with_nerf/settings.py`:
```
LLAVA_PATH = "/workspace/pre-trained-weights/LLaVA/<my_llava_checkpoint>"
```

Open up your directory's permission for the docker container:
```
cd <parent_path_chat-with-nerf>
chmod -R 777 .
```

If using Docker, you can use the following command to spin up a docker container with **chat-with-nerf** mounted under workspace
```bash
docker run --gpus "device=0" -v /<parent_path_chat-with-nerf>/:/workspace/ -v /home/<your_username>/.cache/:/home/user/.cache/ --rm -it --shm-size=12gb chat-with-nerf:latest
```
Then install Chat with NeRF dependencies
```bash
cd /workspace/chat-with-nerf
pip install -e .
pip install -e .[dev]
```
(or use your favorite virtual environment manager)

## :arrow_forward: Inference

### Interactivve Demo

We provide the code to [interactively play](http://sled-whistler.eecs.umich.edu:7777/) with our agent. To run the demo:

```
cd /workspace/chat-with-nerf
export $(cat .env | xargs); gradio chat_with_nerf/app.py
```

### Reproduce Results in the Paper

We provide four Jupyter notebooks to reproduce results in the paper. To run these notebooks, please refer to the [Evaluation README](experiments/evaluation.md).

To facillate easier reproduction of our results, we provide pre-processed data [here](https://www.dropbox.com/scl/fo/2rcihrdlq7zpbteqcg4z0/AFjWxaHrB48ddtFT9BE2j30?rlkey=g9ang0u1h69ppi9g16gj1c5n4&dl=0).

If you would like to use your own 3D scenes, please follow the next two sections:

#### Extracting openscene embeddings

For extracting the openscene embeddings, we used the pre-trained Distillation model checkpoint, shared by the Openscene Authors for generating the representation. To generate the corresponding representations, kindly refer to the guidelines provided in the Openscene GitHub repository, specifically focusing on the Data Preparation and Run Sections.
```
https://github.com/pengsongyou/openscene#data-preparation
https://github.com/pengsongyou/openscene#run
```

#### Extracting LERF embeddings

We include a version of NeRFStudio code in our released docker and you can use generate point cloud function to acquire the H5 embedding. We slightly altered the ns-export function: https://docs.nerf.studio/reference/cli/ns_export.html to get the H5 embeddings.


## Related Work
- [nerfstudio](https://github.com/nerfstudio-project/nerfstudio)
- [LERF](https://github.com/kerrj/lerf)
- [LLaVA](https://github.com/haotian-liu/LLaVA)

## Citation
```
@misc{yang2023llmgrounder,
      title={LLM-Grounder: Open-Vocabulary 3D Visual Grounding with Large Language Model as an Agent}, 
      author={Jianing Yang and Xuweiyi Chen and Shengyi Qian and Nikhil Madaan and Madhavan Iyengar and David F. Fouhey and Joyce Chai},
      year={2023},
      eprint={2309.12311},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
