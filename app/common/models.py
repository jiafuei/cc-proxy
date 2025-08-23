from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class ContentBlock(BaseModel):
    """Base content block model."""

    type: str


class TextContent(ContentBlock):
    """Text content block."""

    type: Literal['text']
    text: str
    cache_control: Optional[Dict[str, Any]] = None


class ThinkingContent(ContentBlock):
    """Thinking content block."""

    type: Literal['thinking']
    thinking: str
    signature: str


class ToolUseContent(ContentBlock):
    """Tool use content block."""

    type: Literal['tool_use']
    id: str
    name: str
    input: Dict[str, Any]


class ToolResultContent(ContentBlock):
    """Tool result content block."""

    type: Literal['tool_result']
    tool_use_id: str
    content: Union[str, List[Dict[str, Any]]]
    is_error: Optional[bool] = None


class ImageSource(BaseModel):
    """Image source model."""

    type: Literal['base64']
    data: str
    media_type: str


class ImageContent(ContentBlock):
    """Image content block."""

    type: Literal['image']
    source: ImageSource


ContentBlockType = Union[TextContent, ThinkingContent, ToolUseContent, ToolResultContent, ImageContent]


class SystemMessage(BaseModel):
    """System message model."""

    type: Literal['text']
    text: str
    cache_control: Optional[Dict[str, Any]] = None


class Message(BaseModel):
    """Message model."""

    role: Literal['user', 'assistant']
    content: Union[List[ContentBlockType], str]


class ToolProperty(BaseModel):
    """Tool property schema."""

    type: str
    description: Optional[str] = None
    default: Optional[Any] = None
    enum: Optional[List[str]] = None
    items: Optional[Dict[str, Any]] = None
    properties: Optional[Dict[str, Any]] = None
    required: Optional[List[str]] = None
    additionalProperties: Optional[bool] = None
    minItems: Optional[int] = None
    minLength: Optional[int] = None


class ToolInputSchema(BaseModel):
    """Tool input schema model."""

    model_config = ConfigDict(validate_by_alias=True)

    type: str
    properties: Dict[str, ToolProperty]
    required: Optional[List[str]] = None
    additionalProperties: bool = False
    schema_: str = Field(alias='$schema', serialization_alias='$schema', default='http://json-schema.org/draft-07/schema#')


class Tool(BaseModel):
    """Tool definition model."""

    name: str
    description: str
    input_schema: ToolInputSchema


class ThinkingConfig(BaseModel):
    """Thinking configuration model."""

    budget_tokens: int
    type: Literal['enabled']


class Metadata(BaseModel):
    """Request metadata model."""

    user_id: str


class ClaudeRequest(BaseModel):
    """Main Claude API request model."""

    model: str
    messages: List[Message]
    temperature: Optional[float] = None
    system: Optional[List[SystemMessage]] = None
    tools: Optional[List[Tool]] = None
    metadata: Optional[Metadata] = None
    max_tokens: int = Field(default=32000)
    thinking: Optional[ThinkingConfig] = None
    stream: Optional[bool] = True

    def to_dict(self):
        return self.model_dump(mode='json', by_alias=True, exclude_none=True)
