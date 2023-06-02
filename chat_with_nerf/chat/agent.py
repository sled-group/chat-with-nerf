import json
import os
import threading
import time
from collections import defaultdict
from collections.abc import Generator

import requests
from requests import Response

from chat_with_nerf import logger
from chat_with_nerf.chat.grounder import ground_with_callback
from chat_with_nerf.chat.session import Session
from chat_with_nerf.model.model_context import ModelContext, ModelContextManager
from chat_with_nerf.settings import Settings
from chat_with_nerf.util import get_status_code_and_reason

if not Settings.USE_FAKE_GROUNDER:
    model_context: ModelContext = ModelContextManager.get_model_context()
else:
    model_context = ModelContext(
        scene_configs=None,  # type: ignore
        visual_grounder=defaultdict(lambda: None),  # type: ignore
        pipeline=None,  # type: ignore
        captioner=None,  # type: ignore
    )  # type: ignore # this model_context is for debugging only

# Streaming endpoint
API_URL = str(os.getenv("API_URL"))
assert API_URL, "API_URL environment variable is not set"

# Huggingface provided GPT4 OpenAI API Key
OPENAI_API_KEY = str(os.getenv("OPENAI_API_KEY"))
assert OPENAI_API_KEY, "OPENAI_API_KEY environment variable is not set"

MAX_ITERATION = 10


# Inferenec function
def act(
    system_msg: str,
    inputs: str,
    top_p: float,
    temperature: float,
    dropdown_scene: str,
    session: Session,
) -> Generator[tuple[list[tuple], int, str | None, Session, str | None], None, None]:
    session.chat_history_for_display.append(
        (inputs, "")
    )  # append in a tuple format, first is user input, second is assistant response
    yield (
        session.chat_history_for_display,
        session.chat_counter,
        None,
        session,
        session.grounding_result_mesh_path,
    )

    retry_sleep_time = 0.5
    give_control_to_user = False
    for _ in range(MAX_ITERATION):  # iterate until GPT decides to give control to user
        if give_control_to_user:
            break

        # exceed free tier limit
        if session.chat_counter >= Settings.MAX_TURNS:
            session.chat_history_for_display.append(
                (
                    None,
                    (
                        f"**SYSTEM: Maximum number of free trial turns ({Settings.MAX_TURNS}) "
                        "reached. Ending dialog.**"
                    ),
                )
            )
            yield (
                session.chat_history_for_display,
                session.chat_counter,
                None,
                session,
                session.grounding_result_mesh_path,
            )
            return

        response: Response | None
        try:
            for returned_session, returned_response in ask_gpt(
                system_msg,
                inputs,
                top_p,
                temperature,
                session,
            ):
                session = returned_session
                response = returned_response
                yield (
                    session.chat_history_for_display,
                    session.chat_counter,
                    get_status_code_and_reason(response),
                    session,
                    session.grounding_result_mesh_path,
                )
                # sometimes GPT returns an empty response because there were too many requests
                # in this case we don't change the chat_history_for_llm, simply re-issue the request
                if response and response.status_code == 429:
                    time.sleep(retry_sleep_time)
                    retry_sleep_time *= 2  # exponential backoff
                    continue
        except ValueError:
            # sometimes GPT returns "The server is currently overloaded with other requests."
            # no need to modify the chat_history_for_llm in this case, simply re-issue the request
            time.sleep(retry_sleep_time)
            retry_sleep_time *= 2  # exponential backoff
            yield (
                session.chat_history_for_display,
                session.chat_counter,
                get_status_code_and_reason(response),
                session,
                session.grounding_result_mesh_path,
            )
            continue

        # done streaming
        try:
            gpt_response_json = json.loads(session.chat_history_for_llm[-1][1])
        except json.decoder.JSONDecodeError as e:
            # if reaches here, then it means GPT-4 indeed returned a non-JSON response
            logger.error(
                f"Cannot decode GPT response: {session.chat_history_for_llm[-1][1]}. "
                "Asking GPT to retry."
            )
            # need to modify the chat_history_for_llm in this case
            # so that the LLM knows it has made a mistake
            inputs = (
                "SYSTEM: The above response caused an error: "
                f"type: {type(e)}, msg: {str(e)}. Please retry."
            )
            # GPT often returns an empty response and causing a JSONDecodeError
            # because there were too many requests,
            # so add some sleep here to avoid spamming GPT
            time.sleep(retry_sleep_time)
            retry_sleep_time *= 2  # exponential backoff
            continue

        try:
            beautified_response_markdown = beautify_gpt_response(gpt_response_json)
            session.chat_history_for_display[-1] = (
                session.chat_history_for_display[-1][0],
                beautified_response_markdown,
            )
            yield (
                session.chat_history_for_display,
                session.chat_counter,
                get_status_code_and_reason(response),
                session,
                session.grounding_result_mesh_path,
            )

            # controller logic to decide what to do next
            if gpt_response_json["command"]["name"] == "user_dialog":
                sentence_to_user = gpt_response_json["command"]["args"][
                    "sentence_to_user"
                ]
                session.chat_history_for_display.append(
                    (None, sentence_to_user)
                )  # use none as user input to display system message only
                give_control_to_user = True
            elif gpt_response_json["command"]["name"] == "ground":
                # first display what GPT wants to tell the user
                session.chat_history_for_display.append(
                    (
                        None,
                        gpt_response_json["thoughts"]["speak"],
                    )
                )
                yield (
                    session.chat_history_for_display,
                    session.chat_counter,
                    get_status_code_and_reason(response),
                    session,
                    session.grounding_result_mesh_path,
                )

                # now start the grounding process
                ground_text = gpt_response_json["command"]["args"]["ground_text"]

                # use a separate thread to do grounding since it takes a while
                grounder_returned_chatbot_msg = None

                def grounding_callback(
                    grounder_results: list[tuple[str, str]], session: Session
                ) -> None:
                    # this function is called when the grounder finishes
                    nonlocal grounder_returned_chatbot_msg, inputs, give_control_to_user
                    chatbot_msg_for_user, pure_text_for_gpt = display_grounder_results(
                        grounder_results, session
                    )
                    inputs = pure_text_for_gpt
                    give_control_to_user = False
                    # this must be last line to ensure thread safety
                    grounder_returned_chatbot_msg = chatbot_msg_for_user

                threading.Thread(
                    target=ground_with_callback,
                    args=(
                        session,
                        dropdown_scene,
                        ground_text,
                        model_context.picture_takers[dropdown_scene],
                        model_context.captioner,
                        grounding_callback,
                    ),
                ).start()

                # while grounder is running, display a loading message
                dot_counter = 0
                first_iteration = True
                while grounder_returned_chatbot_msg is None:
                    dot_counter = (dot_counter + 1) % 4
                    dots = "." * dot_counter
                    if first_iteration:
                        session.chat_history_for_display.append(
                            (None, f"**SYSTEM: I'm thinking{dots}**")
                        )
                        first_iteration = False
                    else:
                        session.chat_history_for_display[-1] = (
                            None,
                            f"**SYSTEM: I'm thinking{dots}**",
                        )
                    yield (
                        session.chat_history_for_display,
                        session.chat_counter,
                        get_status_code_and_reason(response),
                        session,
                        session.grounding_result_mesh_path,
                    )
                    time.sleep(1)  # Adjust the sleep duration as needed

                session.chat_history_for_display.extend(grounder_returned_chatbot_msg)

            elif gpt_response_json["command"]["name"] == "finish_grounding":
                image_ids = gpt_response_json["command"]["args"]["image_id"]
                logger.debug(f"session.image_id_to_path: {session.image_id_to_path}")
                if type(image_ids) == int:
                    img_id_list = [image_ids]
                elif type(image_ids) == str:
                    img_id_list = [int(i) for i in image_ids.split(", ")]

                markdown_to_display = (
                    f"**SYSTEM: Grounding finished. Image id: {image_ids}**\n"
                )

                for img_id in img_id_list:
                    markdown_to_display += (
                        f"![caption](file={session.image_id_to_path[img_id]})"
                    )
                session.chat_history_for_display.append((None, markdown_to_display))
                give_control_to_user = True
            elif gpt_response_json["command"]["name"] == "end_dialog":
                session.chat_history_for_display.append(
                    (None, gpt_response_json["thoughts"]["speak"])
                )
                session.chat_history_for_display.append(
                    (None, "**SYSTEM: End of dialog**")
                )
                give_control_to_user = True

            yield (
                session.chat_history_for_display,
                session.chat_counter,
                get_status_code_and_reason(response),
                session,
                session.grounding_result_mesh_path,
            )

        except Exception as e:
            logger.error(
                f"An error occured: type: {type(e)}, msg: {str(e)}. Asking GPT to retry."
            )
            # need to modify the chat_history_for_llm in this case
            # so that the LLM knows it has made a mistake
            inputs = (
                "SYSTEM: The above response caused an error: "
                f"type: {type(e)}, msg: {str(e)}. Please retry."
            )
            continue

    # update and save session state
    session.save()
    yield (
        session.chat_history_for_display,
        session.chat_counter,
        get_status_code_and_reason(response),
        session,
        session.grounding_result_mesh_path,
    )


def display_grounder_results(
    grounder_results: list[tuple[str, str]],
    session: Session,
) -> tuple[list[tuple[None, str]], str]:
    """Display grounder results in markdown format."""
    str_for_user = ""
    pure_text_for_gpt = ""
    for i, (img_path, caption) in enumerate(grounder_results):
        str_for_user += f"Image {i+1}: {caption}\n ![{caption}](file={img_path})\n"
        pure_text_for_gpt += f"Grounder returned:\nImage {i+1}: {caption}\n"
        # record the image_id to image path mapping in session
        session.image_id_to_path[i + 1] = img_path
    logger.info(f"pure_text_for_gpt: {pure_text_for_gpt}")
    chatbot_msg_for_user = [(None, str_for_user)]

    return chatbot_msg_for_user, pure_text_for_gpt


def ask_gpt(
    system_msg: str,
    inputs: str,
    top_p: float,
    temperature: float,
    session: Session,
) -> Generator[tuple[Session, Response | None], None, None]:
    headers = {
        "Content-Type": "application/json",
        "api-key": OPENAI_API_KEY,
    }

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

    if session.chat_counter == 0:
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
    else:  # if chat_counter != 0 :
        messages = multi_turn_message  # Of the type: [{"role": "system", "content": system_msg},]
        for data in session.chat_history_for_llm:
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

    session.chat_counter += 1

    session.chat_history_for_llm.append(
        (inputs, "")
    )  # append in a tuple format, first is user input, second is assistant response
    if session.chat_history_for_display[-1][0] is None:
        # if the last turn doesn't have a user input (this is after grounder returns),
        # then give a new empty turn to session.chat_history_for_display to work with
        # otherwise, it would delete the last turn
        session.chat_history_for_display.append((None, None))
    logger.info(f"Logging : payload is - {payload}")
    # make a POST request to the API endpoint using the requests.post method, passing in stream=True
    response = requests.post(API_URL, headers=headers, json=payload, stream=True)
    logger.info(f"Logging : response code - {response}")
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
            if chunk.startswith("error:"):
                # sometimes GPT returns "The server is currently overloaded with other requests."
                msg = f"Received error from API: {chunk}"
                logger.error(msg)
                raise ValueError(msg)
            elif (
                len(chunk) > 12
                and chunk.startswith("data:")
                and "content" in json.loads(chunk[6:])["choices"][0]["delta"]
            ):
                partial_words = (
                    partial_words
                    + json.loads(chunk[6:])["choices"][0]["delta"]["content"]
                )

                session.chat_history_for_llm[-1] = (
                    session.chat_history_for_llm[-1][0],
                    partial_words,
                )
                session.chat_history_for_display[-1] = (
                    session.chat_history_for_display[-1][0],
                    partial_words,
                )

                token_counter += 1
                yield session, response

    yield session, response


def beautify_gpt_response(gpt_response_json) -> str:
    # beautify the response
    beautified_response_markdown = "# **Agent Reasoning Summary**\n"
    beautified_response_markdown += (
        f"- Observation:\n {gpt_response_json['thoughts']['observation']}\n\n"
    )
    beautified_response_markdown += (
        f"- Reasoning:\n {gpt_response_json['thoughts']['reasoning']}\n\n"
    )
    beautified_response_markdown += (
        f"- Plan:\n {gpt_response_json['thoughts']['plan']}\n\n"
    )
    beautified_response_markdown += (
        f"- Self-critique:\n {gpt_response_json['thoughts']['self-critique']}\n\n"
    )
    beautified_response_markdown += (
        f"- Speak:\n {gpt_response_json['thoughts']['speak']}\n\n"
    )
    return beautified_response_markdown
