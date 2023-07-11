import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Annotated

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request, status, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from sse_starlette.sse import EventSourceResponse

from chatbot import from_bot_map_config, ChatBotBase
from openai_object import (
    ChatCompletionResponseStreamChoice, DeltaMessage, ChatCompletionResponse, ChatCompletionRequest,
    ChatCompletionResponseChoice, ChatMessage, ModelList, ModelCard
)
from util import load_config

try:
    import torch
except (ImportError, ModuleNotFoundError):
    torch = None

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

config = load_config(os.environ['CONFIG_FILE'])
bot_map: Dict[str, ChatBotBase] = {}
model_list: ModelList = ModelList(data=[ModelCard(id=bot_name) for bot_name in config['bot_map'].keys()])
token_list: list = config['token_list']


@asynccontextmanager
async def lifespan(_):  # collects GPU memory
    yield
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


app = FastAPI(lifespan=lifespan)
logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.exception_handler(Exception)
async def exception_handler(request: Request, e: Exception) -> Response:
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": f"{e.__class__.__name__}: {str(e)}",
            "status": 500
        }
    )


@app.post('/api/chat/{model}')
async def chat(model: str, request: Request, token: Annotated[str | None, Header()]):
    if token not in token_list:
        raise HTTPException(status_code=401, detail="Invalid API key")
    json_request: Dict[str, Any] = await request.json()
    query: str = json_request['query']
    history: List[List[str]] = json_request.get('history', [])
    parameters: Dict[str, Any] = json_request.get('parameters', {})
    response = bot_map[model].chat(query, history=history, parameters=parameters or {})
    return {
        "response": response,
        "history": history + [[query, response]],
        "status": 200
    }


@app.websocket("/api/chat/{model}/ws")
async def steam_chat(websocket: WebSocket, model: str, token: Annotated[str | None, Header()]):
    """
    input: JSON String of {"query": "", "history": []}
    output: JSON String of {"response": "", "history": [], "status": 200}
        status 200 stand for response ended, else not
    """
    if token not in token_list:
        raise HTTPException(status_code=401, detail="Invalid API key")
    await websocket.accept()
    try:
        while True:
            json_request: dict = await websocket.receive_json()
            query: str = json_request['query']
            history: list = json_request['history']
            parameters: dict = json_request.get('parameters', {})
            logger.info(f'{websocket.client.host}:{websocket.client.port} query = {query}')
            for response in bot_map[model].stream_chat(query, history=history, parameters=parameters):
                await websocket.send_json({
                    "response": response,
                    "history": history + [[query, response]],
                    "status": 202
                })
            await websocket.send_json({"status": 200})
    except WebSocketDisconnect:
        pass


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    return model_list


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest, authorization: Annotated[str | None, Header()]):
    if not (authorization.startswith('Bearer ') and authorization.replace('Bearer ', '') in token_list):
        raise HTTPException(status_code=401, detail="Invalid API key")
    if request.messages[-1].role != "user":
        raise HTTPException(status_code=400, detail="Invalid request")
    query = request.messages[-1].content

    prev_messages = request.messages[:-1]
    system = prev_messages.pop(0).content if len(prev_messages) > 0 and prev_messages[0].role == "system" else None

    history = []
    assert len(prev_messages) % 2 == 0, f'context length should be even, got {len(prev_messages)}'
    for i in range(0, len(prev_messages), 2):
        if prev_messages[i].role == "user" and prev_messages[i + 1].role == "assistant":
            history.append([prev_messages[i].content, prev_messages[i + 1].content])

    parameters = {}

    if request.stream:
        generate = stream_chat(query, history, system, parameters, request.model)
        return EventSourceResponse(generate, media_type="text/event-stream")

    response = bot_map[request.model].chat(query, history, system, parameters)
    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role="assistant", content=response),
        finish_reason="stop"
    )

    return ChatCompletionResponse(model=request.model, choices=[choice_data], object="chat.completion")


async def stream_chat(query: str, history: List[List[str]], system: str, parameters: dict, model_id: str):
    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(role="assistant"),
        finish_reason=None
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.json(exclude_unset=True, ensure_ascii=False))

    current_length = 0

    for new_response in bot_map[model_id].stream_chat(query, history=history, system=system, parameters=parameters):
        if len(new_response) == current_length:
            continue

        new_text = new_response[current_length:]
        current_length = len(new_response)

        choice_data = ChatCompletionResponseStreamChoice(
            index=0,
            delta=DeltaMessage(content=new_text),
            finish_reason=None
        )
        chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
        yield "{}".format(chunk.json(exclude_unset=True, ensure_ascii=False))

    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(),
        finish_reason="stop"
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.json(exclude_unset=True, ensure_ascii=False))
    yield '[DONE]'


def main():
    for k, v in from_bot_map_config(config['bot_map']).items():
        bot_map[k] = v
    uvicorn.run(
        'main:app',
        host=config.get('server', {}).get('host', '0.0.0.0'),
        port=config.get('server', {}).get('port', 8000),
        workers=config.get('server', {}).get('workers', None)
    )


if __name__ == '__main__':
    main()
