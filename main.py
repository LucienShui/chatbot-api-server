import logging
import os
from typing import Any, Dict, List, Annotated

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request, status, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from sse_starlette.sse import EventSourceResponse

from chatbot import from_bot_map_config, ChatBotBase, Converter
from openai_object import (
    ChatCompletionResponseStreamChoice, DeltaMessage, ChatCompletionResponse, ChatCompletionRequest,
    ChatCompletionResponseChoice, ChatMessage, ModelList, ModelCard
)
from util import load_config, logger

try:
    import torch
except (ImportError, ModuleNotFoundError):
    torch = None

config = load_config(os.environ.get('CONFIG_FILE', 'config.json'))
bot_map: Dict[str, ChatBotBase] = {}
model_list: ModelList = ModelList(data=[])
token_list: list = config['token_list']

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
async def exception_handler(_: Request, e: Exception) -> Response:
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": f"{e.__class__.__name__}: {str(e)}",
            "status": 500
        }
    )


@app.on_event('startup')
def init():
    for k, v in from_bot_map_config(config['bot_map'], config.get('alias', {}), config.get('disable', [])).items():
        model_list.data.append(ModelCard(id=k))
        bot_map[k] = v


@app.post('/api/chat/{model}')
async def chat(model: str, request: Request, token: Annotated[str | None, Header()]):
    if token not in token_list:
        raise HTTPException(status_code=401, detail="Invalid API key")
    json_request: Dict[str, Any] = await request.json()
    query: str = json_request['query']
    history: List[List[str]] = json_request.get('history', [])
    system: str = json_request.get('system', None)
    parameters: Dict[str, Any] = json_request.get('parameters', {})
    response = bot_map[model].chat(Converter.to_messages(query, history, system), parameters=parameters or {})
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
            system: str = json_request.get('system', None)
            logger.info(f'{websocket.client.host}:{websocket.client.port} query = {query}')
            for response in bot_map[model].stream_chat(
                    Converter.to_messages(query, history, system), parameters=parameters or {}):
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
    parameters: dict = {}
    for k in ['temperature', 'top_p', 'max_length']:
        v = getattr(request, k)
        if v is not None:
            parameters[k] = v
    messages = [message.__dict__ for message in request.messages]

    if request.stream:
        generate = stream_chat(messages, parameters, request.model)
        return EventSourceResponse(generate, media_type="text/event-stream")

    response = bot_map[request.model].chat(messages, parameters)
    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role="assistant", content=response),
        finish_reason="stop"
    )

    return ChatCompletionResponse(model=request.model, choices=[choice_data], object="chat.completion")


async def stream_chat(messages: List[Dict[str, str]], parameters: dict, model_id: str):
    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(role="assistant"),
        finish_reason=None
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.json(exclude_unset=True, ensure_ascii=False))

    current_length = 0

    for new_response in bot_map[model_id].stream_chat(messages, parameters=parameters):
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
