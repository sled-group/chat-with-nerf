# Adapted from https://huggingface.co/spaces/ysharma/ChatGPT4

import gradio as gr

from nerf_grounding_chat_interface import agent
from nerf_grounding_chat_interface.system_prompt import DEFAULT_SYSTEM_PROMPT


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

with gr.Blocks(
    css=(
        "#col_container { margin-left: auto; margin-right: auto;} #chatbot {height: 520px; "
        "overflow: auto;}"
    ),
    # theme=theme,
) as demo:
    gr.HTML(title)

    with gr.Column(elem_id="col_container"):
        # GPT4 API Key is provided by Huggingface
        with gr.Accordion(label="System message:", open=False):
            system_msg = gr.Textbox(
                label="Instruct the AI Assistant to set its beaviour",
                info=system_msg_info,
                value=default_system_msg,
            )
            accordion_msg = gr.HTML(
                value="🚧 To set System message you will have to refresh the app",
                visible=False,
            )
        with gr.Row():
            with gr.Column(scale=5):
                model_3d = gr.Model3D(
                    value="/workspace/nerf-grounding-chat-interface/3d_asset/poly.glb",
                    clear_color=[0.0, 0.0, 0.0, 0.0],
                    label="3D Model",
                )
            with gr.Column(scale=5):
                chatbot_for_display = gr.Chatbot(label="GPT4", elem_id="chatbot")

        inputs = gr.Textbox(
            placeholder="Hi there!", label="Type an input and press Enter"
        )
        gpt_chat_state = gr.State([])
        with gr.Row():
            with gr.Column(scale=7):
                send_button = gr.Button("Send").style(full_width=True)
            with gr.Column(scale=3):
                server_status_code = gr.Textbox(
                    label="Status code from OpenAI server",
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
            chat_counter = gr.Number(value=0, visible=False, precision=0)

    # Event handling
    inputs.submit(
        agent.act,
        [
            system_msg,
            inputs,
            top_p,
            temperature,
            chat_counter,
            gpt_chat_state,
            chatbot_for_display,
        ],
        [chatbot_for_display, gpt_chat_state, chat_counter, server_status_code],
    )  # openai_api_key
    send_button.click(
        agent.act,
        [
            system_msg,
            inputs,
            top_p,
            temperature,
            chat_counter,
            gpt_chat_state,
            chatbot_for_display,
        ],
        [chatbot_for_display, gpt_chat_state, chat_counter, server_status_code],
    )  # openai_api_key

    inputs.submit(set_interactive_false, [], [system_msg])
    send_button.click(set_interactive_false, [], [system_msg])
    inputs.submit(set_visible_true, [], [accordion_msg])
    send_button.click(set_visible_true, [], [accordion_msg])

    send_button.click(reset_textbox, [], [inputs])
    inputs.submit(reset_textbox, [], [inputs])

    # Examples
    with gr.Accordion(label="Examples for user message:", open=False):
        gr.Examples(
            examples=[
                ["I want to sit down."],
                ["yellow sofa"],
                ["I want something to drink"],
            ],
            inputs=inputs,
        )

gr.close_all()
demo.queue(max_size=99, concurrency_count=20).launch(
    debug=True, server_name="0.0.0.0", server_port=7777
)
