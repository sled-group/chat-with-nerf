# :sauropod: Chat with NeRF

## :bulb: Highlight

- **Open-Vocabulary 3D Localization.** Localize **everything** with language!
- **Dynamic Grounding.** Humans will be able to chat with agent to localize novel objects.

## :fire: News

## :star: Explanations/Tips for Chat with NeRF Inputs and Outputs

## :label: TODO

## :hammer_and_wrench: Install

To install the dependencies we provide a Dockerfile:
```bash
docker build -t chat-with-nerf:latest .
```

Then locally you need to run
```bash
git clone https://github.com/sled-group/chat-with-nerf.git
```
Run the Docker using the following fashion, especially mount chat-with-nerf under workspace
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

To run the demo:

```
cd /workspace/chat-with-nerf
export $(cat .env | xargs); gradio chat_with_nerf/app.py
```
