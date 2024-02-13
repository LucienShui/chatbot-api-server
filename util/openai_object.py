import time
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field


class Role:
    user = "user"
    assistant = "assistant"
    system = "system"
    literal = Literal["user", "assistant", "system"]


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "owner"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = None


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []


class ChatMessage(BaseModel):
    role: Role.literal
    content: str


class DeltaMessage(BaseModel):
    role: Optional[Role.literal] = None
    content: Optional[str] = None


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length"]


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]]


class ChatCompletionUsage(BaseModel):
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]


class ChatCompletionResponse(BaseModel):
    model: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: List[Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))
    usage: Optional[ChatCompletionUsage] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    api_type: Optional[str] = None
    api_version: Optional[str] = None
    engine: Optional[str] = None

    def response(self, content: str = None, usage: ChatCompletionUsage = None) -> ChatCompletionResponse:
        stream_choice, response = ChatCompletionResponseStreamChoice, ChatCompletionResponse  # alias for short
        if self.stream:
            if content is not None:  # streaming
                choice = stream_choice(index=0, delta=DeltaMessage(content=content), finish_reason=None)
            elif usage is None:  # before streaming
                choice = stream_choice(index=0, delta=DeltaMessage(role=Role.assistant), finish_reason=None)
            else:  # after streaming
                choice = stream_choice(index=0, delta=DeltaMessage(), finish_reason="stop")
            return response(model=self.model, choices=[choice], usage=usage, object="chat.completion.chunk")
        else:
            message = ChatMessage(role=Role.assistant, content=content)
            choice = ChatCompletionResponseChoice(index=0, message=message, finish_reason="stop")
            return response(model=self.model, choices=[choice], usage=usage, object="chat.completion")
