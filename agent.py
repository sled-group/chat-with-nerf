import os
import json
from typing import List, Tuple
import requests
from grounder import ground

# Streaming endpoint
API_URL = os.getenv("API_URL")

# Huggingface provided GPT4 OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

MAX_ITERATION = 10


# Inferenec function
def act(
    system_msg: str,
    inputs: str,
    top_p,
    temperature,
    chat_counter,
    gpt_chat_state=[],
    chatbot_for_display=[],
):
    give_control_to_user = False
    for _ in range(MAX_ITERATION):  # iterate until GPT decides to give control to user
        if give_control_to_user:
            break

        for chatbot_for_display, gpt_chat_state, chat_counter, response in ask_gpt(
            system_msg,
            inputs,
            top_p,
            temperature,
            chat_counter,
            gpt_chat_state,
            chatbot_for_display,
        ):
            yield chatbot_for_display, gpt_chat_state, chat_counter, response

        # done streaming
        try:
            gpt_response_json = json.loads(chatbot_for_display[-1][1])
        except json.decoder.JSONDecodeError:
            inputs = "SYSTEM: The above response caused an error: json.decoder.JSONDecodeError"
            continue

        beautified_response_markdown = beautify_gpt_response(gpt_response_json)
        chatbot_for_display.append((None, beautified_response_markdown))
        yield chatbot_for_display, gpt_chat_state, chat_counter, response

        # controller logic to decide what to do next
        if gpt_response_json["command"]["name"] == "user_dialog":
            sentence_to_user = gpt_response_json["command"]["args"]["sentence_to_user"]
            chatbot_for_display.append(
                (None, sentence_to_user)
            )  # use none as user input to display system message only
            give_control_to_user = True
        elif gpt_response_json["command"]["name"] == "ground":
            ground_text = gpt_response_json["command"]["args"]["ground_text"]
            grounder_results = ground(ground_text)
            markdown_for_user, pure_text_for_gpt = display_grounder_results(
                grounder_results
            )
            chatbot_for_display.append(
                (None, markdown_for_user)
            )  # use none as user input to display system message only
            # after grounder returns, give the results to GPT
            # and let GPT decide what to do next
            inputs = pure_text_for_gpt
            give_control_to_user = False
        elif gpt_response_json["command"]["name"] == "finish_grounding":
            image_id = gpt_response_json["command"]["args"]["image_id"]
            chatbot_for_display.append(
                (None, f"SYSTEM: Grounding finished. Image id: {image_id}")
            )
            give_control_to_user = True
        elif gpt_response_json["command"]["name"] == "end_dialog":
            chatbot_for_display.append((None, "SYSTEM: End of dialog"))
            give_control_to_user = True

        yield chatbot_for_display, gpt_chat_state, chat_counter, response


def display_grounder_results(
    grounder_results: List[Tuple[str, str]]
) -> Tuple[str, str]:
    """Display grounder results in markdown format"""
    markdown_for_user = ""
    pure_text_for_gpt = ""
    for i, (img_path, caption) in enumerate(grounder_results):
        markdown_for_user += f"![{caption}]({img_path})\nImage {i+1}: {caption}\n"
        pure_text_for_gpt += f"Grounder returned:\nImage {i+1}: {caption}\n"
    return markdown_for_user, pure_text_for_gpt


def ask_gpt(
    system_msg,
    inputs,
    top_p,
    temperature,
    chat_counter,
    gpt_chat_state=[],
    chatbot_for_display=[],
):
    headers = {
        "Content-Type": "application/json",
        "api-key": OPENAI_API_KEY,
    }
    print(f"system message is ^^ {system_msg}")

    if system_msg.strip() == "":
        initial_message = [
            {"role": "user", "content": f"{inputs}"},
        ]
        multi_turn_message = []
    else:
        initial_message = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"{inputs}"},
        ]
        multi_turn_message = [
            {"role": "system", "content": system_msg},
        ]

    if chat_counter == 0:
        payload = {
            "model": "gpt-4",
            "messages": initial_message,
            "temperature": 1.0,
            "top_p": 1.0,
            "n": 1,
            "stream": True,
            "presence_penalty": 0,
            "frequency_penalty": 0,
        }
        print(f"chat_counter - {chat_counter}")
    else:  # if chat_counter != 0 :
        messages = multi_turn_message  # Of the type of - [{"role": "system", "content": system_msg},]
        for data in gpt_chat_state:
            user = {}
            user["role"] = "user"
            user["content"] = data[0]
            assistant = {}
            assistant["role"] = "assistant"
            assistant["content"] = data[1]
            messages.append(user)
            messages.append(assistant)
        temp = {}
        temp["role"] = "user"
        temp["content"] = inputs
        messages.append(temp)
        # messages
        payload = {
            "model": "gpt-4",
            "messages": messages,  # Of the type of [{"role": "user", "content": f"{inputs}"}],
            "temperature": temperature,  # 1.0,
            "top_p": top_p,  # 1.0,
            "n": 1,
            "stream": True,
            "presence_penalty": 0,
            "frequency_penalty": 0,
        }

    chat_counter += 1

    gpt_chat_state.append(
        (inputs, "")
    )  # append in a tuple format, first is user input, second is assistant response
    chatbot_for_display.append(
        (inputs, "")
    )  # append in a tuple format, first is user input, second is assistant response
    print(f"Logging : payload is - {payload}")
    # make a POST request to the API endpoint using the requests.post method, passing in stream=True
    response = requests.post(API_URL, headers=headers, json=payload, stream=True)
    print(f"Logging : response code - {response}")
    token_counter = 0
    partial_words = ""

    counter = 0
    for chunk in response.iter_lines():
        # Skipping first chunk
        if counter == 0:
            counter += 1
            continue
        # check whether each line is non-empty
        if chunk.decode():
            chunk = chunk.decode()
            # decode each line as response data is in bytes
            if (
                len(chunk) > 12
                and "content" in json.loads(chunk[6:])["choices"][0]["delta"]
            ):
                partial_words = (
                    partial_words
                    + json.loads(chunk[6:])["choices"][0]["delta"]["content"]
                )

                gpt_chat_state[-1] = (gpt_chat_state[-1][0], " " + partial_words)
                chatbot_for_display[-1] = (
                    chatbot_for_display[-1][0],
                    " " + partial_words,
                )

                token_counter += 1
                yield chatbot_for_display, gpt_chat_state, chat_counter, response

    return chatbot_for_display, gpt_chat_state, chat_counter, response


def beautify_gpt_response(gpt_response_json) -> str:
    # beautify the response
    beautified_response_markdown = "#### Agent Reasoning Summary ####\n"
    beautified_response_markdown += (
        f"Text:\n {gpt_response_json['thoughts']['text']}\n\n"
    )
    beautified_response_markdown += (
        f"Reasoning:\n {gpt_response_json['thoughts']['reasoning']}\n\n"
    )
    beautified_response_markdown += (
        f"Plan:\n {gpt_response_json['thoughts']['reasoning']}\n\n"
    )
    beautified_response_markdown += (
        f"Criticism:\n {gpt_response_json['thoughts']['criticism']}\n\n"
    )
    beautified_response_markdown += (
        f"Speak:\n {gpt_response_json['thoughts']['speak']}\n\n"
    )
    return beautified_response_markdown
