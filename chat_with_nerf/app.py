# Adapted from https://huggingface.co/spaces/ysharma/ChatGPT4

import os
from signal import SIGTERM
from time import sleep

import gradio as gr
from psutil import process_iter

from chat_with_nerf.chat import agent
from chat_with_nerf.chat.session import Session
from chat_with_nerf.chat.system_prompt import DEFAULT_SYSTEM_PROMPT
from chat_with_nerf.settings import Settings
from chat_with_nerf.util import list_dirs


# Resetting to blank
def reset_textbox():
    return gr.update(value="")


# to set a component as visible=False
def set_visible_false():
    return gr.update(visible=False)


# to set a component as visible=True
def set_visible_true():
    return gr.update(visible=True)


def set_interactive_false():
    return gr.update(interactive=False)


def change_scene(
    dropdown_scene: str,
) -> tuple[str, str | None, list, int, str | None, Session]:
    # reset model_3d, chatbot_for_display, chat_counter, server_status_code
    new_session = Session.create_for_scene(dropdown_scene)
    return (
        os.path.join(Settings.data_path, dropdown_scene, "poly.glb"),
        None,
        new_session.chat_history_for_display,
        new_session.chat_counter,
        None,
        new_session,
    )


title = """<h1 align="center">🔥 Chat with NeRF using GPT-4 🚀</h1>
<p><center>
<a href="https://chat-with-nerf.github.io/" target="_blank">[Project Page]</a>
<a href="https://github.com/sled-group/chat-with-nerf" target="_blank">[Code]</a>
</center></p>
"""

# <div style="background-color:yellow;color:white;padding:2%;">
#     <center><strong style="color:black;">
#         👋🏻 Note: Sometimes system response might be slow due to popularity of the GPT-4 API.
#     </strong></center>
#     <center><strong style="color:black;">
#         If you encounter the error message "maximum number of free
#         trial turns reached", please refresh the page and retry.
#         We appreciate your understanding and patience! 🙏
#     </strong></center>
# </div>


# Using info to add additional information about System message in GPT4
system_msg_info = (
    "A conversation could begin with a system message to gently instruct the assistant. "
    "System message helps set the behavior of the AI Assistant. "
    "For example, the assistant could be instructed with 'You are a helpful assistant."
)

default_system_msg = DEFAULT_SYSTEM_PROMPT

# Modifying existing Gradio Theme
theme = gr.themes.Soft(
    primary_hue="zinc",
    secondary_hue="green",
    neutral_hue="green",
    text_size=gr.themes.sizes.text_lg,
)

with gr.Blocks() as demo:
    gr.HTML(title)

    session_state = gr.State(Session.create)
    with gr.Column():
        with gr.Row():
            with gr.Column(scale=5):
                # GPT4 API Key is provided by Huggingface
                dropdown_scene = gr.Dropdown(
                    choices=list_dirs(Settings.data_path),
                    value="home_1",
                    interactive=True,
                    label="Select a scene",
                )
                model_3d = gr.Model3D(
                    value=Settings.data_path + "/home_1" + "/poly.glb",
                    clear_color=[0.0, 0.0, 0.0, 0.0],
                    label="3D Model",
                )
                gr.HTML(
                    """<center><strong>
                    👆 SCROLL or DRAG on the 3D Model
                    to zoom in/out and rotate. Press CTRL and DRAG to pan.
                    </strong></center>
                    """
                )
                gr.HTML(
                    """<center><strong>
                    👇 When grounding finishes,
                    a green bounding sphere will appear around the object of interest.
                    </strong></center>
                    """
                )
                model_3d_grounding_result = gr.Model3D(
                    clear_color=[0.0, 0.0, 0.0, 0.0],
                    label="Grounding Result",
                )
            with gr.Column(scale=5):
                with gr.Row():
                    # openai_api_key = gr.Textbox(
                    #     label=(
                    #         "Paste your OpenAI API key here and press Enter↵ "
                    #         "or leave emtpy for free trial"
                    #     ),
                    #     type="password",
                    # )
                    chat_counter = gr.Textbox(
                        value=0,
                        label=f"Turn count (free trial limit: {Settings.MAX_TURNS})",
                    )
                    server_status_code = gr.Textbox(
                        label="Status code from OpenAI server", interactive=False
                    )
                chat_history_for_display = gr.Chatbot(
                    value=[(None, Settings.INITIAL_MSG_FOR_DISPLAY)],
                    label="Chat Assistant",
                ).style(height="600")
                with gr.Row():
                    with gr.Column(scale=8):
                        user_chat_input = gr.Textbox(
                            placeholder="I want to find the chair near the table",
                            show_label=False,
                        )
                    with gr.Column(scale=1, min_width=0):
                        send_button = gr.Button("Send", variant="primary").style(
                            full_width=True
                        )
                    with gr.Column(scale=1, min_width=0):
                        clear_button = gr.Button("Clear").style(full_width=True)
                with gr.Row():
                    # Examples
                    with gr.Accordion(label="Examples for user message:", open=True):
                        gr.Examples(
                            examples=[
                                ["How many doors are there in this room?"],
                                ["Find the chair near the table."],
                                ["Where is the fridge?"],
                                ["a white plate on a red square-shaped cutting board"],
                                ["I am hungry, can you find me something to eat?"],
                                ["这里一共有几扇门？"],
                                ["この部屋にはドアが何枚ありますか?"],
                            ],
                            inputs=user_chat_input,
                        )

        with gr.Accordion(label="System instruction:", open=False, visible=False):
            system_msg = gr.Textbox(
                label="Instruct the AI Assistant to set its beaviour",
                info=system_msg_info,
                value=default_system_msg,
            )
            accordion_msg = gr.HTML(
                value="🚧 To set System message you will have to refresh the app",
                visible=False,
            )
        # top_p, temperature
        with gr.Accordion("Parameters", open=False, visible=False):
            top_p = gr.Slider(
                minimum=-0,
                maximum=1.0,
                value=1.0,
                step=0.05,
                interactive=True,
                label="Top-p (nucleus sampling)",
            )
            temperature = gr.Slider(
                minimum=-0,
                maximum=5.0,
                value=1.0,
                step=0.1,
                interactive=True,
                label="Temperature",
            )
    gr.Markdown("### Terms of Service")
    gr.HTML(
        """By using this service, users are required to agree to the following terms:
The service is a research preview intended for non-commercial use only.
The service may collect user dialogue data for future research."""
    )

    # Event handling
    dropdown_scene.change(
        fn=change_scene,
        inputs=[dropdown_scene],
        outputs=[
            model_3d,
            model_3d_grounding_result,
            chat_history_for_display,
            chat_counter,
            server_status_code,
            session_state,
        ],
    )
    clear_button.click(
        fn=change_scene,
        inputs=[dropdown_scene],
        outputs=[
            model_3d,
            model_3d_grounding_result,
            chat_history_for_display,
            chat_counter,
            server_status_code,
            session_state,
        ],
    )
    user_chat_input.submit(
        fn=agent.act,
        inputs=[
            system_msg,
            user_chat_input,
            top_p,
            temperature,
            dropdown_scene,
            session_state,
        ],
        outputs=[
            chat_history_for_display,
            chat_counter,
            server_status_code,
            session_state,
            model_3d_grounding_result,
        ],
    )  # openai_api_key
    send_button.click(
        fn=agent.act,
        inputs=[
            system_msg,
            user_chat_input,
            top_p,
            temperature,
            dropdown_scene,
            session_state,
        ],
        outputs=[
            chat_history_for_display,
            chat_counter,
            server_status_code,
            session_state,
            model_3d_grounding_result,
        ],
    )  # openai_api_key

    user_chat_input.submit(set_interactive_false, [], [system_msg])
    send_button.click(set_interactive_false, [], [system_msg])
    user_chat_input.submit(set_visible_true, [], [accordion_msg])
    send_button.click(set_visible_true, [], [accordion_msg])

    send_button.click(reset_textbox, [], [user_chat_input])
    user_chat_input.submit(reset_textbox, [], [user_chat_input])


sleep_time = 2
port = 7777
for x in range(1, 8):  # try 8 times
    try:
        # put your logic here
        gr.close_all()
        demo.queue(
            max_size=20,
            concurrency_count=3,
            api_open=False,
        ).launch(
            debug=True,
            server_name="0.0.0.0",
            server_port=port,
        )
    except OSError:
        for proc in process_iter():
            for conns in proc.connections(kind="inet"):
                if conns.laddr.port == port:
                    proc.send_signal(SIGTERM)  # or SIGKILL
        print(f"Retrying {x} time...")
        pass

    sleep(sleep_time)  # wait for 2 seconds before trying to fetch the data again
    sleep_time *= 2  # exponential backoff
