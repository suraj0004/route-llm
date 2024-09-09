from collections import defaultdict
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Optional

import pandas as pd
from tqdm import tqdm

from routellm.routers.routers import ROUTER_CLS
import logging
import aiohttp
import json

from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()   

# Access the environment variable
llm_queue_gateway_base_path = os.getenv('LLM_QUEUE_GATEWAY_BASE_PATH')


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
    **kwargs,
    ):
        logging.info(kwargs)
        
        if "model" in kwargs:
            router, threshold = self._parse_model_name(kwargs["model"])

        self._validate_router_threshold(router, threshold)
        model_type = self._get_routed_model_for_completion(kwargs["messages"], router, threshold)

        if model_type == 'strong':
            kwargs["model"] = 'mistral'
        else:
            kwargs["model"] = 'tinyllama'

        # Construct the full URL by appending the full_path
        url = f'{llm_queue_gateway_base_path}/{full_path}'
        
        logging.info(f"Requesting URL: {url}")

        # Use aiohttp for asynchronous HTTP requests
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=kwargs) as response:
                response.raise_for_status()

                # Handle streaming progress and direct streaming back to the client
                if response.status == 200:
                    # Streaming progress
                    if kwargs['stream'] == True:
                        # Stream each chunk back to the client as soon as it is received
                        async for chunk in response.content.iter_any():
                            yield chunk
                            
                    elif response.content_type == 'application/x-ndjson':
                        # Handle NDJSON streaming
                        async for line in response.content:
                            line = line.decode('utf-8').strip()
                            if line:
                                yield f"{line}\n".encode('utf-8')
                    else:
                        # Handle standard JSON responses
                        data = await response.json()
                        yield json.dumps(data).encode('utf-8')
