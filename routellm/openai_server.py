"""A server that provides OpenAI-compatible RESTful APIs.

It current only supports Chat Completions: https://platform.openai.com/docs/api-reference/chat)
"""

import argparse
import logging
import os
import time
from collections import defaultdict
from typing import AsyncGenerator, Dict, List, Literal, Optional, Union

import fastapi
import shortuuid
import uvicorn
import yaml
from fastapi.concurrency import asynccontextmanager
from fastapi.responses import JSONResponse, StreamingResponse
# from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from routellm.controller import Controller, RoutingError
from routellm.routers.routers import ROUTER_CLS
from fastapi import Request, HTTPException

from typing import AsyncIterable, List
from fastapi.responses import StreamingResponse
import json
from typing import AsyncIterator

os.environ["TOKENIZERS_PARALLELISM"] = "false"
CONTROLLER = None

# openai_client = AsyncOpenAI()
count = defaultdict(lambda: defaultdict(int))


@asynccontextmanager
async def lifespan(app):
    global CONTROLLER

    CONTROLLER = Controller(
        routers=args.routers,
        config=yaml.safe_load(open(args.config, "r")) if args.config else None,
        strong_model=args.strong_model,
        weak_model=args.weak_model,
        api_base=args.base_url,
        api_key=args.api_key,
        progress_bar=True,
    )
    yield
    CONTROLLER = None


app = fastapi.FastAPI(lifespan=lifespan)


class ErrorResponse(BaseModel):
    object: str = "error"
    message: str


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0


class ChatCompletionRequest(BaseModel):
    # OpenAI fields: https://platform.openai.com/docs/api-reference/chat/create
    model: str
    messages: Union[
        str,
        List[Dict[str, str]],
        List[Dict[str, Union[str, List[Dict[str, Union[str, Dict[str, str]]]]]]],
    ]
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Dict[int, float]] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    max_tokens: Optional[int] = None
    n: Optional[int] = None
    presence_penalty: Optional[float] = None
    response_format: Optional[Dict[str, str]] = (
        None  # { "type": "json_object" } for json mode
    )
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    tools: Optional[List[Dict[str, Union[str, int, float]]]] = None
    tool_choice: Optional[str] = None
    user: Optional[str] = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[Literal["stop", "length"]] = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{shortuuid.random()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo


# async def stream_response(response) -> AsyncGenerator:
#     async for chunk in response:
#         yield f"data: {chunk.model_dump_json()}\n\n"
#     yield "data: [DONE]\n\n"

async def stream_response(response: List[dict]) -> AsyncIterable[bytes]:
    for item in response:
        # Convert each item to a JSON string and then to bytes
        yield json.dumps(item).encode('utf-8')
        # If there should be a delay between chunks, you can use `await asyncio.sleep(seconds)`

# @app.post("/route-llm/{full_path:path}")
# async def create_chat_completion(request: Request, full_path: str):
#     # The model name field contains the parameters for routing.
#     # Model name uses format router-[router name]-[threshold] e.g. router-bert-0.7
#     # The router type and threshold is used for routing that specific request.

#     # The full_path variable will capture anything after '/route-llm/'
#     logging.info(f"Received request at /route-llm/{full_path}")

#     body = await request.json()
#     logging.info(f"Received request: {body}")
#     try:
#         stream = body.get("stream", True)  # Determine if streaming is required from request body
#         body["stream"] = stream
#         logging.info(stream)
#         if stream == True:
#             async def generate_stream() -> AsyncIterator[bytes]:
#                 async for line in CONTROLLER.acompletion( full_path=full_path, **body):
#                     yield line

#             return StreamingResponse(
#                 generate_stream(),
#                 media_type="application/x-ndjson",
#                 headers={"Connection": "keep-alive"}
#             )
#         else:
#             logging.info("json")
#             async for response_data in CONTROLLER.acompletion( full_path=full_path, **body):
#                 return JSONResponse(content=json.loads(response_data.decode('utf-8')))
#     except Exception as e:
#         logging.error(f"Error: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

@app.post("/route-llm/{qengine_workflow_id}/{full_path:path}")
async def create_chat_completion(request: Request, qengine_workflow_id: str, full_path: str):
    logging.info(f"Received request at /route-llm/{qengine_workflow_id}/{full_path}")

    try:
        # Parse the JSON body from the request
        body = await request.json()
        logging.info(f"Request body: {body}")

        # Determine if streaming is required
        stream = body.get("stream", True)
        body["stream"] = stream
        logging.info(f"Streaming enabled: {stream}")

        # Handle streaming response
        if stream:
            logging.info("Preparing streaming response")

            async def generate_stream() -> AsyncIterator[bytes]:
                async for chunk in CONTROLLER.acompletion(full_path=full_path, qengine_workflow_id=qengine_workflow_id, **body):
                    yield chunk

            return StreamingResponse(
                generate_stream(),
                media_type="application/x-ndjson",
                headers={"Connection": "keep-alive"}
            )

        else:
            # Handle non-streaming response
            logging.info("Handling non-streaming response")
            async for response_data in CONTROLLER.acompletion(full_path=full_path, qengine_workflow_id=qengine_workflow_id, **body):
                return JSONResponse(content=json.loads(response_data.decode('utf-8')))

    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse request body as JSON: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    except Exception as e:
        logging.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")



@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return JSONResponse(content={"status": "online"})

@app.get("/")
async def welcome():
    """welcome and threshold list endpoint."""
    return JSONResponse(content={"status": "200", "message":"server is up and running", "mf-routers-with-threshold": {
        "router-mf-0.62589":"0%",
        "router-mf-0.24034":"10%",
        "router-mf-0.1881":"20%",
        "router-mf-0.15609":"30%",
        "router-mf-0.1339":"40%",
        "router-mf-0.11593":"50%",
        "router-mf-0.09995":"60%",
        "router-mf-0.08535":"70%",
        "router-mf-0.07075":"80%",
        "router-mf-0.05438":"90%",
        "router-mf-0.00937":"100%",
    }})

parser = argparse.ArgumentParser(
    description="An OpenAI-compatible API server for LLM routing."
)
parser.add_argument(
    "--verbose",
    action="store_true",
)
parser.add_argument("--workers", type=int, default=0)
parser.add_argument("--config", type=str, default=None)
parser.add_argument("--port", type=int, default=6060)
parser.add_argument(
    "--routers",
    nargs="+",
    type=str,
    default=["mf"],
    choices=list(ROUTER_CLS.keys()),
)
parser.add_argument(
    "--base-url",
    help="The base URL used for all LLM requests",
    type=str,
    default=None,
)
parser.add_argument(
    "--api-key",
    help="The API key used for all LLM requests",
    type=str,
    default=None,
)
parser.add_argument("--strong-model", type=str, default="strong")
parser.add_argument(
    "--weak-model", type=str, default="weak"
)
args = parser.parse_args()

if args.verbose:
    logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    print("Launching server with routers:", args.routers)
    uvicorn.run(
        "routellm.openai_server:app",
        port=args.port,
        host="0.0.0.0",
        workers=args.workers,
    )
