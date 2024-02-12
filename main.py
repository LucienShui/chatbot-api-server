import os
import time
from typing import Any, Dict, List, Annotated, Iterable

import uvicorn
from fastapi import FastAPI, HTTPException, Request, status, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from sse_starlette.sse import EventSourceResponse

from model.base import ChatAPIBase
from model.loader import from_bot_map_config
from util.openai_object import (ChatCompletionResponse, ChatCompletionRequest, ChatCompletionResponseChoice,
                                ChatMessage, ModelList, ModelCard)
from util.logger import logger
from util import load_config

config = load_config(os.environ.get('CONFIG_FILE', 'config.json'))
bot_map: Dict[str, ChatAPIBase] = {}
model_list: ModelList = ModelList(data=[])
token_list: list = config['token_list']

app = FastAPI()

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


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    return model_list


def stream_wrapper(request: ChatCompletionRequest) -> Iterable:
    start_time = time.time()
    response = ''
    chunk = None
    for chunk in bot_map[request.model].chat(request):
        if delta := chunk.choices[0].delta.content:
            response += delta
        yield "{}".format(chunk.json(exclude_unset=True))

    logger.info({'method': f'/v1/chat/completions', 'request': request.dict(exclude_unset=True), 'response': response,
                 'usage': chunk.usage.dict(exclude_unset=True) if chunk else {},
                 'cost': f'{(time.time() - start_time) * 1000:.2f} ms'})


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest, authorization: Annotated[str | None, Header()]):
    if not (authorization.startswith('Bearer ') and authorization.replace('Bearer ', '') in token_list):
        raise HTTPException(status_code=401, detail="Invalid API key")

    if request.stream:
        generate = stream_wrapper(request)
        return EventSourceResponse(generate, media_type="text/event-stream")
    else:
        start_time = time.time()
        bot = bot_map[request.model]
        completion_response = next(bot.chat(request))
        response = completion_response.choices[0].message.content
        usage = completion_response.usage
        logger.info({'method': f'/v1/chat/completions', 'request': request.dict(exclude_unset=True),
                     'response': response, 'usage': usage.dict(exclude_unset=True),
                     'cost': f'{(time.time() - start_time) * 1000:.2f} ms'})

    return completion_response


def main():
    uvicorn.run(
        'main:app',
        host=config.get('server', {}).get('host', '0.0.0.0'),
        port=config.get('server', {}).get('port', 8000),
        workers=config.get('server', {}).get('workers', None)
    )


if __name__ == '__main__':
    main()
