import logging

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from transformers import AutoTokenizer, AutoModel

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

pretrained = "THUDM/chatglm-6b-int4-qe"
tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)
model = AutoModel.from_pretrained(pretrained, trust_remote_code=True).half().cuda()
model = model.eval()
app = FastAPI()
logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware
)

with open('index.html') as f:
    html = f.read()


@app.get("/")
async def get():
    return HTMLResponse(html)


@app.post("/api/chat/")
async def chat(request: Request):
    json_request: dict = await request.json()
    query = json_request['query']
    history = json_request['history']
    logger.info(f'{request.client.host}:{request.client.port} query = {query}')
    response, history = model.chat(tokenizer, query, history=history)
    return {
        "response": response,
        "history": history,
        "status": 200
    }


@app.websocket("/api/chat/ws")
async def steam_chat(websocket: WebSocket):
    """
    input: JSON String of {"query": "", "history": []}
    output: JSON String of {"response": "", "history": [], "status": 200}
        status 200 stand for response ended, else not
    """
    await websocket.accept()
    try:
        while True:
            json_request = await websocket.receive_json()
            query = json_request['query']
            history = json_request['history']
            logger.info(f'{websocket.client.host}:{websocket.client.port} query = {query}')
            for response, history in model.stream_chat(tokenizer, query, history=history):
                await websocket.send_json({
                    "response": response,
                    "history": history,
                    "status": 202
                })
            await websocket.send_json({"status": 200})
    except WebSocketDisconnect:
        pass


def main():
    uvicorn.run(f"{__name__}:app", host='0.0.0.0', port=8000, workers=1)


if __name__ == '__main__':
    main()
