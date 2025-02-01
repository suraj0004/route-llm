from collections import defaultdict
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Optional
from datetime import datetime
import pandas as pd
from tqdm import tqdm

from routellm.routers.routers import ROUTER_CLS
import logging
import aiohttp
import json

from dotenv import load_dotenv
import os
from routellm.ollama_request_parser import OllamaRequestParser
from routellm.parsers.OpenAIToOllamaParser import OpenAIToOllamaParser
# Load environment variables from the .env file
load_dotenv()   

# Access the environment variable
# llm_queue_gateway_base_path = os.getenv('LLM_QUEUE_GATEWAY_BASE_PATH')
openai_api_key = os.getenv('OPENAI_API_KEY')
Q_ENGINE_API_KEY = os.getenv('Q_ENGINE_API_KEY')
Q_ENGINE_BASE_PATH = os.getenv('Q_ENGINE_BASE_PATH')
# Default config for routers augmented using golden label data from GPT-4.
# This is exactly the same as config.example.yaml.
GPT_4_AUGMENTED_CONFIG = {
    "mf": {"checkpoint_path": "routellm/mf_gpt4_augmented"},
}


class RoutingError(Exception):
    pass


@dataclass
class ModelPair:
    strong: str
    weak: str


class Controller:
    def __init__(
        self,
        routers: list[str],
        strong_model: str,
        weak_model: str,
        config: Optional[dict[str, dict[str, Any]]] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        progress_bar: bool = False,
    ):
        self.model_pair = ModelPair(strong=strong_model, weak=weak_model)
        self.routers = {}
        self.api_base = api_base
        self.api_key = api_key
        self.model_counts = defaultdict(lambda: defaultdict(int))
        self.progress_bar = progress_bar

        if config is None:
            config = GPT_4_AUGMENTED_CONFIG

        router_pbar = None
        if progress_bar:
            router_pbar = tqdm(routers)
            tqdm.pandas()

        for router in routers:
            if router_pbar is not None:
                router_pbar.set_description(f"Loading {router}")
            self.routers[router] = ROUTER_CLS[router](**config.get(router, {}))

        # Some Python magic to match the OpenAI Python SDK
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=self.acompletion, acreate=self.acompletion
            )
        )

        self.ollama_parser = OllamaRequestParser()
        self.openai_parser = OpenAIToOllamaParser()

    def _validate_router_threshold(
        self, router: Optional[str], threshold: Optional[float]
    ):
        if router is None or threshold is None:
            raise RoutingError("Router or threshold unspecified.")
        if router not in self.routers:
            raise RoutingError(
                f"Invalid router {router}. Available routers are {list(self.routers.keys())}."
            )
        if not 0 <= threshold <= 1:
            raise RoutingError(
                f"Invalid threshold {threshold}. Threshold must be a float between 0.0 and 1.0."
            )

    def _parse_model_name(self, model: str):
        _, router, threshold = model.split("-", 2)
        try:
            threshold = float(threshold)
        except ValueError as e:
            raise RoutingError(f"Threshold {threshold} must be a float.") from e
        if not model.startswith("router"):
            raise RoutingError(
                f"Invalid model {model}. Model name must be of the format 'router-[router name]-[threshold]."
            )
        return router, threshold

    def _get_routed_model_for_completion(
        self, messages: list, router: str, threshold: float
    ):
        # Look at the last turn for routing.
        # Our current routers were only trained on first turn data, so more research is required here.
        prompt = messages[-1]["content"]
        routed_model = self.routers[router].route(prompt, threshold, self.model_pair)

        self.model_counts[router][routed_model] += 1

        return routed_model

    # Mainly used for evaluations
    def batch_calculate_win_rate(
        self,
        prompts: pd.Series,
        router: str,
    ):
        self._validate_router_threshold(router, 0)
        router_instance = self.routers[router]
        if router_instance.NO_PARALLEL and self.progress_bar:
            return prompts.progress_apply(router_instance.calculate_strong_win_rate)
        elif router_instance.NO_PARALLEL:
            return prompts.apply(router_instance.calculate_strong_win_rate)
        else:
            return prompts.parallel_apply(router_instance.calculate_strong_win_rate)

    def route(self, prompt: str, router: str, threshold: float):
        self._validate_router_threshold(router, threshold)

        return self.routers[router].route(prompt, threshold, self.model_pair)

    # Matches OpenAI's Chat Completions interface, but also supports optional router and threshold args
    # If model name is present, attempt to parse router and threshold using it, otherwise, use the router and threshold args
    # def completion(
    #     self,
    #     *,
    #     router: Optional[str] = None,
    #     threshold: Optional[float] = None,
    #     **kwargs,
    # ):
    #     if "model" in kwargs:
    #         router, threshold = self._parse_model_name(kwargs["model"])

    #     self._validate_router_threshold(router, threshold)
    #     kwargs["model"] = self._get_routed_model_for_completion(
    #         kwargs["messages"], router, threshold
    #     )
    #     return completion(api_base=self.api_base, api_key=self.api_key, **kwargs)

    # Matches OpenAI's Async Chat Completions interface, but also supports optional router and threshold args
    # async def acompletion(
    #     self,
    #     *,
    #     router: Optional[str] = None,
    #     threshold: Optional[float] = None,
    #     full_path: str,
    #     **kwargs,
    # ):
    #     logging.info(kwargs)
    #     if "model" in kwargs:
    #         router, threshold = self._parse_model_name(kwargs["model"])

    #     self._validate_router_threshold(router, threshold)
    #     model_type = self._get_routed_model_for_completion(
    #         kwargs["messages"], router, threshold
    #     )

    #     logging.info(f"model type: {model_type}")
    #     if model_type == 'strong':
    #         kwargs["model"] = 'mistral'
    #     else:
    #         kwargs["model"] = 'tinyllama'

 
    #     logging.info('__________________________model name:__________________________')
    #     logging.info(kwargs)

    #      # Construct the full URL by appending the full_path
    #     url = f'http://host.docker.internal:8070/llm-queue-gateway/{full_path}'
        
    #     logging.info(f"Requesting URL: {url}")

    #     # Use aiohttp for asynchronous HTTP requests
    #     async with aiohttp.ClientSession() as session:
    #         async with session.post(
    #             url,
    #             json=kwargs
    #         ) as response:
    #             response.raise_for_status()

    #             if kwargs['stream'] == True:
    #                 # Handle Event Stream (text/event-stream) streaming
    #                 async for chunk in response.content:
    #                     chunk = chunk.decode('utf-8').strip()
    #                     if chunk:
    #                         yield f"{chunk}\n".encode('utf-8')

                # elif response.content_type == 'application/x-ndjson':
                #     # Handle NDJSON streaming
                #     async for line in response.content:
                #         line = line.decode('utf-8').strip()
                #         if line:
                #             yield f"{line}\n".encode('utf-8')

    #             else:
    #                 # Handle standard JSON responses
    #                 data = await response.json()
    #                 yield json.dumps(data).encode('utf-8')
      
    #     # return await acompletion(api_base=self.api_base, api_key=self.api_key, **kwargs)


    async def acompletion(
        self,
        *,
        router: Optional[str] = None,
        threshold: Optional[float] = None,
        full_path: str,
        qengine_workflow_id: str,
        **kwargs,
    ):
        try:
            input_payload = kwargs
            input_api = full_path
            logging.info('Starting acompletion with input_payload: %s', input_payload)
            logging.info(f"input_api: {input_api}")

            # Parse model name if provided
            if "model" in input_payload:
                router, threshold = self._parse_model_name(input_payload["model"])

            # Validate router and threshold
            self._validate_router_threshold(router, threshold)
            
            # Determine model type based on the router and threshold
            model_type = self._get_routed_model_for_completion(input_payload.get("messages"), router, threshold)

            logging.info('Calculated model type: %s', model_type)
            strong_model = 'gpt-4'
            weak_model = 'gpt-3.5-turbo'
            async with aiohttp.ClientSession() as session:
                async with session.get(f'{Q_ENGINE_BASE_PATH}/api/studio/workflow/{qengine_workflow_id}/llm-models?apiKey={Q_ENGINE_API_KEY}') as response:
                    response.raise_for_status()
                    logging.info(f"Response status: {response.status}")

                    # Parse the JSON response into a dictionary
                    data = await response.json()
                    logging.info(data)

                    # Check if the 'success' key exists and is True
                    if data.get("success") == True:  # Use .get() to avoid KeyError if 'success' is missing
                        strong_model = data["data"]["strongModel"]
                        weak_model = data["data"]["weakModel"]  # Correctly access 'weakModel'
                    else:
                        logging.error("The 'success' key is not True or missing in the response.")


            # Set the appropriate model
            input_payload["model"] = strong_model if model_type == 'strong' else weak_model
            logging.info('Model name: %s', input_payload["model"])

            # Construct the full URL by appending the full_path
            
            # url = f'{llm_queue_gateway_base_path}/{full_path}'
            logging.info(f"\n\n Parsing ollama payload to openai \n\n ")
            logging.info(f"before parsing : {json.dumps(input_payload, indent=2)} \n\n ")
            parsed_response =  self.ollama_parser.parse_request(input_api, input_payload,"openai")
            logging.info(f"after parsing: {json.dumps(parsed_response, indent=2)} \n\n ")
            headers = {
                    "Authorization": f"Bearer {openai_api_key}",
                    "Content-Type": "application/json"
                }
            logging.info(f"Requesting URL: {parsed_response['url']} \n\n ")
            logging.info(f"headers: {headers} \n\n ")
            logging.info(f"body: {parsed_response['body']} \n\n ")

            # Asynchronous HTTP request with error handling
            async with aiohttp.ClientSession() as session:
                
                async with session.post(parsed_response["url"], json=parsed_response["body"], headers=headers) as response:
                    response.raise_for_status()
                    logging.info(f"Response status: {response.status}")

                    # Handle different response types (streaming or standard JSON)
                    if input_payload.get('stream', False):
                        logging.info('Streaming response enabled')

                        async for chunk in response.content.iter_any():
                            # Call the parse_chunk function from the OpenAItoOllamaResponseParser class
                            for ollama_chunk in self.openai_parser.parse_chunk(chunk):
                                # Yield or process the Ollama chunk
                                yield ollama_chunk
                        #   yield chunk
                    elif response.content_type == 'application/x-ndjson':
                        logging.info('NDJSON streaming detected')
                        async for line in response.content:
                            line = line.decode('utf-8').strip()
                            if line:
                                yield f"{line}\n".encode('utf-8')
                    else:
                    # Standard JSON response
                        logging.info('Processing standard JSON response')
                        data = await response.json()
                        logging.info(f" \n\n response from openai: \n\n {json.dumps(data, indent=2)}  \n\n ")
                        yield json.dumps(self.openai_parser.parse(data)).encode('utf-8')

        except aiohttp.ClientError as e:
            logging.error(e)
            logging.error('HTTP request failed: %s', e)
            raise RuntimeError(f'Failed to request {parsed_response["url"]}') from e
        except Exception as e:
            logging.error(e)
            logging.error('Error during acompletion: %s', e)
            raise RuntimeError('An error occurred during acompletion') from e