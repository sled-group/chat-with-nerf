import os
import threading
import time
import attr
from collections import defaultdict
from collections.abc import Generator
import numpy as np
import requests
import json5
import traceback
from requests import Response
from chat_with_nerf.chat.util import fix_brackets
from chat_with_nerf import logger
from chat_with_nerf.chat.grounder import (
    ground_with_callback,
    ground_no_gpt_with_callback,
    ground_with_callback_with_gpt,
    highlight_clusters_in_mesh,
)
from chat_with_nerf.chat.session import Session
from chat_with_nerf.model.model_context import ModelContext, ModelContextManager
from chat_with_nerf.settings import Settings
from chat_with_nerf.util import get_status_code_and_reason


@attr.define
class Agent:
    model_context: ModelContext = attr.field(init=False, default=None)
    API_URL: str = attr.field(default=str(os.getenv("API_URL")))
    OPENAI_API_KEY: str = attr.field(default=str(os.getenv("OPENAI_API_KEY")))
    MAX_ITERATION: int = 10
    scene_name: str = attr.field(default="scene0025_00")

    def __attrs_post_init__(self):
        # Check for fake grounder
        if self.scene_name is None:
            raise ValueError("default scene_name is not set")

        if not Settings.USE_FAKE_GROUNDER:
            if Settings.NO_GPT:
                self.model_context = ModelContextManager.get_model_no_gpt_context(
                    self.scene_name
                )
            else:
                self.model_context = (
                    ModelContextManager.get_model_no_visual_feedback_openscene_context()
                )
        else:
            self.model_context = ModelContext(
                scene_configs=None,
                visual_grounder=defaultdict(lambda: None),
                pipeline=None,
                captioner=None,
            )

        # Raise exceptions for required environment variables
        if not self.API_URL:
            raise ValueError("API_URL environment variable is not set")

        if not self.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

    def display_grounder_results(
        self, grounder_results: dict, session: Session
    ) -> tuple[list[tuple[None, str]], str]:
        """Display grounder results in markdown format."""
        str_for_user = str(grounder_results)
        pure_text_for_gpt = str(grounder_results)
        # for i, (img_path, caption) in enumerate(grounder_results):
        #     str_for_user += f"Image {i+1}: {caption}\n ![{caption}](file={img_path})\n"
        #     pure_text_for_gpt += f"Grounder returned:\nImage {i+1}: {caption}\n"
        #     # record the image_id to image path mapping in session
        #     session.image_id_to_path[i + 1] = img_path
        logger.info(f"pure_text_for_gpt: {pure_text_for_gpt}")
        chatbot_msg_for_user = [(None, None)]
        # chatbot_msg_for_user = [(None, str_for_user)]
        return chatbot_msg_for_user, pure_text_for_gpt

    def ask_gpt(
        self,
        system_msg: str,
        inputs: str,
        top_p: float,
        temperature: float,
        session: Session,
    ) -> Generator[tuple[Session, Response | None], None, None]:
        headers = {
            "Content-Type": "application/json",
            "api-key": self.OPENAI_API_KEY,
        }
        # find base mesh path
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
        response = requests.post(
            self.API_URL, headers=headers, json=payload, stream=True
        )
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
                    and "content" in json5.loads(chunk[6:])["choices"][0]["delta"]
                ):
                    partial_words = (
                        partial_words
                        + json5.loads(chunk[6:])["choices"][0]["delta"]["content"]
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

    @staticmethod
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

    def act_no_gpt(
        self,
        inputs: str,
        dropdown_scene: str,
        session: Session,
    ):
        # now start the grounding process
        ground_text = inputs

        # use a separate thread to do grounding since it takes a while
        bbox = None

        def grounding_callback(
            grounder_results: np.ndarray | None, session: Session
        ) -> None:
            # this function is called when the grounder finishes
            nonlocal bbox

            # TODO: grounder_results can be None, we should handle it
            if grounder_results is None:
                logger.error("No Bounding Box returned.")
                return

            bbox = grounder_results

        threading.Thread(
            target=ground_no_gpt_with_callback,
            args=(
                session,
                ground_text,
                self.model_context.picture_takers[dropdown_scene],
                grounding_callback,
            ),
        ).start()

        # while grounder is running, display a loading message
        while bbox is None:
            time.sleep(1)  # Adjust the sleep duration as needed

        return bbox

    # Inferenec function
    def act(
        self,
        system_msg: str,
        inputs: str,
        top_p: float,
        temperature: float,
        dropdown_scene: str,
        session: Session,
    ) -> Generator[
        tuple[list[tuple], int, str | None, Session, str | None], None, None
    ]:
        session.base_mesh_path = self.model_context.scene_configs[
            dropdown_scene
        ].load_mesh
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
        session.working_scene_name = dropdown_scene
        retry_sleep_time = 0.1
        give_control_to_user = False
        for _ in range(
            self.MAX_ITERATION
        ):  # iterate until GPT decides to give control to user
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
                for returned_session, returned_response in self.ask_gpt(
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
            except ValueError as exp:
                # sometimes GPT returns "The server is currently overloaded with other requests."
                # no need to modify the chat_history_for_llm in this case, simply re-issue the request
                print(exp)
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
                gpt_response_json_fixed = fix_brackets(
                    session.chat_history_for_llm[-1][1]
                )
                gpt_response_json = json5.loads(gpt_response_json_fixed)
            except Exception as exp:
                # if reaches here, then it means GPT-4 indeed returned a non-JSON response
                logger.error(
                    f"Cannot decode GPT response: {session.chat_history_for_llm[-1][1]}. "
                    f"Asking GPT to retry. Exception: {exp}"
                )
                inputs = (
                    "SYSTEM: The above response caused an error: "
                    f"type: {type(exp)}, msg: {str(exp)}. Please retry."
                )
                # GPT often returns an empty response and causing a JSONDecodeError
                # because there were too many requests,
                # so add some sleep here to avoid spamming GPT
                time.sleep(retry_sleep_time)
                retry_sleep_time *= 2  # exponential backoff
                continue

            try:
                beautified_response_markdown = Agent.beautify_gpt_response(
                    gpt_response_json
                )
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
                print("gpt_response_json: ", gpt_response_json)
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

                    ground_json = gpt_response_json["command"]["args"]["ground_json"]
                    print("ground text: ", ground_json)
                    session.grounding_query = ground_json["target"]["phrase"]
                    # use a separate thread to do grounding since it takes a while
                    grounder_returned_chatbot_msg = None

                    def grounding_callback(
                        grounder_results: list[tuple[str, str] | None],
                        session: Session,
                    ) -> None:
                        # this function is called when the grounder finishes
                        nonlocal grounder_returned_chatbot_msg, inputs, give_control_to_user

                        # TODO: grounder_results can be None, we should handle it
                        if grounder_results is None:
                            inputs = "Visual grounder did not return any images."
                            return

                        (
                            chatbot_msg_for_user,
                            pure_text_for_gpt,
                        ) = self.display_grounder_results(
                            grounder_results, session  # type: ignore
                        )
                        inputs = pure_text_for_gpt
                        give_control_to_user = False
                        print("$" * 100)
                        print(inputs)
                        print("$" * 100)
                        # this must be last line to ensure thread safety
                        grounder_returned_chatbot_msg = chatbot_msg_for_user

                    thread = threading.Thread(
                        target=ground_with_callback_with_gpt,
                        args=(
                            session,
                            dropdown_scene,
                            ground_json,
                            self.model_context.picture_takers[dropdown_scene],
                            self.model_context.captioner,
                            grounding_callback,
                        ),
                    )
                    thread.start()

                    # while grounder is running, display a loading message
                    dot_counter = 0
                    first_iteration = True
                    while grounder_returned_chatbot_msg is None:
                        if thread.is_alive():
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
                        else:
                            raise ValueError(
                                "Grounding thread exited before callback was called."
                            )

                    session.chat_history_for_display.extend(
                        grounder_returned_chatbot_msg
                    )

                elif gpt_response_json["command"]["name"] == "finish_grounding":
                    top_1_object_id = gpt_response_json["command"]["args"][
                        "top_1_object_id"
                    ]
                    top_5_object2scores = gpt_response_json["command"]["args"][
                        "top_5_objects_scores"
                    ]
                    # logger.debug(
                    #     f"session.image_id_to_path: {session.image_id_to_path}"
                    # )
                    if isinstance(top_1_object_id, int):
                        img_id_list = [top_1_object_id]
                    elif isinstance(top_1_object_id, str):
                        img_id_list = [int(i) for i in top_1_object_id.split(", ")]
                    elif isinstance(top_1_object_id, list):
                        img_id_list = top_1_object_id

                    assert len(img_id_list) == 1
                    session.chosen_candidate_id = img_id_list[0]
                    session.top_5_objects2scores = top_5_object2scores

                    # TODO: draw bounding box and mesh right here
                    mesh_file_path = highlight_clusters_in_mesh(
                        session, self.model_context.picture_takers[dropdown_scene].mesh
                    )
                    session.grounding_result_mesh_path = mesh_file_path
                    ## TODO: how to get the correct mesh display?
                    if not session.working_scene_name.startswith("s"):
                        path2images = self.model_context.picture_takers[
                            dropdown_scene
                        ].take_picture_for_the_ground_result(session, img_id_list[0])
                        markdown_to_display = ""
                        markdown_to_display += (
                            " **Top Candidates with corresponding novel view: **  \n"
                        )

                        for i, path2image in enumerate(path2images):
                            markdown_to_display += (
                                f" ![caption](file={path2image.rgb_address}) \n\n"
                            )

                        last_message = f"**SYSTEM: Grounding finished. Ground Object id: {img_id_list[0]}**\n"
                        session.chat_history_for_display.append((None, last_message))
                        session.chat_history_for_display.append(
                            (None, markdown_to_display)
                        )
                        give_control_to_user = True
                    else:
                        last_message = f"**SYSTEM: Grounding finished. Ground Object id: {img_id_list[0]}**\n"
                        session.chat_history_for_display.append((None, last_message))
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

            except Exception as exp:
                logger.error(
                    f"An error occured: type: {type(exp)}, msg: {traceback.format_exc()}. Please check."
                )
                logger.error(
                    f"An error occured: type: {type(exp)}, msg: {str(exp)}. Asking GPT to retry."
                )
                # need to modify the chat_history_for_llm in this case
                # so that the LLM knows it has made a mistake
                inputs = (
                    "SYSTEM: The above response caused an error: "
                    f"type: {type(exp)}, msg: {str(exp)}. Please retry."
                )
                continue

        yield (
            session.chat_history_for_display,
            session.chat_counter,
            get_status_code_and_reason(response),
            session,
            session.grounding_result_mesh_path,
        )

        return (
            session.chat_history_for_display,
            session.chat_counter,
            get_status_code_and_reason(response),
            session,
            session.grounding_result_mesh_path,
        )
