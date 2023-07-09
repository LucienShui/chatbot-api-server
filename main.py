import logging
import os
from typing import Dict, List, Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response

from openai_object import ChatCompletionResponseStreamChoice, DeltaMessage, ChatCompletionResponse, \
    ChatCompletionRequest, ChatCompletionResponseChoice, ChatMessage, ModelList, ModelCard
from util import load_config
from chatbot import from_bot_map_config, ChatBotBase
import time
import torch
import uvicorn
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Literal, Optional, Union
from transformers import AutoTokenizer, AutoModel
from sse_starlette.sse import ServerSentEvent, EventSourceResponse

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

config = load_config(os.environ['CONFIG_FILE'])
bot_map: Dict[str, ChatBotBase] = from_bot_map_config(config['bot_map'])
model_list: ModelList = ModelList(data=[ModelCard(id=bot_name) for bot_name in bot_map.keys()])


app = FastAPI()
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
async def chat(model: str, request: Request):
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
async def steam_chat(websocket: WebSocket, model: str):
    """
    input: JSON String of {"query": "", "history": []}
    output: JSON String of {"response": "", "history": [], "status": 200}
        status 200 stand for response ended, else not
    """
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
async def create_chat_completion(request: ChatCompletionRequest):
    if request.messages[-1].role != "user":
        raise HTTPException(status_code=400, detail="Invalid request")
    query = request.messages[-1].content

    prev_messages = request.messages[:-1]
    system = prev_messages.pop(0).content if len(prev_messages) > 0 and prev_messages[0].role == "system" else None

    history = []
    assert len(prev_messages) % 2 != 0, f'context length should be even, got {len(prev_messages)}'
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
    uvicorn.run(
        'main:app',
        host=config.get('server', {}).get('host', '0.0.0.0'),
        port=config.get('server', {}).get('port', 8000),
        workers=config.get('server', {}).get('workers', None)
    )


if __name__ == '__main__':
    main()
