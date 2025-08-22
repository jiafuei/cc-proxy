from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class Message(BaseModel):
    role: str
    content: str


class MessagesRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: Optional[bool] = True
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stop: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class StreamChunk(BaseModel):
    id: Optional[str] = None
    type: Optional[str] = None
    delta: Optional[Dict[str, Any]] = None
