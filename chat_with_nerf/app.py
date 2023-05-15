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
) -> tuple[str, list, int, str | None, Session]:
    # reset model_3d, chatbot_for_display, chat_counter, server_status_code
    new_session = Session.create_for_scene(dropdown_scene)
    return (
        os.path.join(Settings.data_path, dropdown_scene, "poly.glb"),
        new_session.chat_history_for_display,
        new_session.chat_counter,
        None,
        new_session,
    )


title = """<h1 align="center">🔥Chat with NeRF using GPT-4🚀</h1>"""


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

    with gr.Column():
        with gr.Row():
            openai_api_key = gr.Textbox(
                label="Paste your OpenAI API key here and press Enter↵",
                type="password",
            )
            server_status_code = gr.Textbox(
                label="Status code from OpenAI server", interactive=False
            )
        with gr.Row():
            with gr.Column(scale=5):
                # GPT4 API Key is provided by Huggingface
                dropdown_scene = gr.Dropdown(
                    choices=list_dirs(Settings.data_path),
                    value="office",
                    interactive=True,
                    label="Select a scene",
                )
                model_3d = gr.Model3D(
                    value=Settings.data_path + "/office" + "/poly.glb",
                    clear_color=[0.0, 0.0, 0.0, 0.0],
                    label="3D Model",
                )
            with gr.Column(scale=5):
                chat_history_for_display = gr.Chatbot(
                    value=[(None, Settings.INITIAL_MSG_FOR_DISPLAY)],
                    label="Chat Assistant",
                ).style(height="600")
                with gr.Row():
                    with gr.Column(scale=9):
                        user_chat_input = gr.Textbox(
                            placeholder="I want to find a cutting board",
                            show_label=False,
                        )
                    with gr.Column(scale=1, min_width=0):
                        send_button = gr.Button("Send").style(full_width=True)
        session_state = gr.State(Session.create)

        # Examples
        with gr.Accordion(label="Examples for user message:", open=True):
            gr.Examples(
                examples=[
                    ["I want to sit down."],
                    ["yellow sofa"],
                    ["I want something to drink"],
                ],
                inputs=user_chat_input,
            )
        with gr.Accordion(label="System instruction:", open=False):
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
        with gr.Accordion("Parameters", open=False):
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
            chat_counter = gr.Number(
                value=0, visible=True, precision=0, label="Turn count"
            )

    # Event handling
    dropdown_scene.change(
        fn=change_scene,
        inputs=[dropdown_scene],
        outputs=[
            model_3d,
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
        demo.queue(max_size=99, concurrency_count=20).launch(
            debug=True, server_name="0.0.0.0", server_port=port
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
