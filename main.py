import logging
import os
from typing import Dict, List, Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from util import load_config
from chatbot import from_bot_map_config, ChatBotBase

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

app = FastAPI()
logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware
)

with open('index.html') as f:
    html = f.read()


@app.exception_handler(Exception)
async def exception_handler(request: Request, e: Exception) -> Response:
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": f"{e.__class__.__name__}: {str(e)}",
            "status": 500
        }
    )


@app.get("/")
async def get() -> Response:
    return HTMLResponse(html)


@app.post('/api/chat/{bot}')
async def chat(bot: str, request: Request):
    json_request: Dict[str, Any] = await request.json()
    query: str = json_request['query']
    history: List[List[str]] = json_request.get('history', [])
    parameters: Dict[str, Any] = json_request.get('parameters', {})
    response = bot_map[bot].chat(query, history=history, parameters=parameters or {})
    return {
        "response": response,
        "history": history + [[query, response]],
        "status": 200
    }


@app.websocket("/api/chat/{bot}/ws")
async def steam_chat(websocket: WebSocket, bot: str):
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
            for response in bot_map[bot].stream_chat(query, history=history, parameters=parameters):
                await websocket.send_json({
                    "response": response,
                    "history": history + [[query, response]],
                    "status": 202
                })
            await websocket.send_json({"status": 200})
    except WebSocketDisconnect:
        pass


def main():
    uvicorn.run(
        'main:app',
        host=config.get('server', {}).get('host', '0.0.0.0'),
        port=config.get('server', {}).get('port', 8000),
        workers=config.get('server', {}).get('workers', None)
    )


if __name__ == '__main__':
    main()
