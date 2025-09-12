from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class AnthropicContentBlock(BaseModel):
    """Base content block model."""

    model_config = ConfigDict(extra='allow')

    type: str


class AnthropicTextContent(AnthropicContentBlock):
    """Text content block."""

    model_config = ConfigDict(extra='allow')

    type: Literal['text']
    text: str
    cache_control: Optional[Dict[str, Any]] = None


class AnthropicThinkingContent(AnthropicContentBlock):
    """Thinking content block."""

    type: Literal['thinking']
    thinking: str
    signature: str


class AnthropicToolUseContent(AnthropicContentBlock):
    """Tool use content block."""

    model_config = ConfigDict(extra='allow')

    type: Literal['tool_use']
    id: str
    name: str
    input: Dict[str, Any]


class AnthropicToolResultContent(AnthropicContentBlock):
    """Tool result content block."""

    model_config = ConfigDict(extra='allow')

    type: Literal['tool_result']
    tool_use_id: str
    content: Union[str, List[Dict[str, Any]]]
    is_error: Optional[bool] = None


class AnthropicImageSource(BaseModel):
    """Image source model."""

    model_config = ConfigDict(extra='allow')

    type: Literal['base64']
    data: str
    media_type: str


class AnthropicImageContent(AnthropicContentBlock):
    """Image content block."""

    model_config = ConfigDict(extra='allow')

    type: Literal['image']
    source: AnthropicImageSource


AnthropicContentBlockType = Union[AnthropicTextContent, AnthropicThinkingContent, AnthropicToolUseContent, AnthropicToolResultContent, AnthropicImageContent]


class AnthropicSystemMessage(BaseModel):
    """System message model."""

    model_config = ConfigDict(extra='allow')

    type: Literal['text']
    text: str
    cache_control: Optional[Dict[str, Any]] = None


class AnthropicMessage(BaseModel):
    """Message model."""

    model_config = ConfigDict(extra='allow')

    role: Literal['user', 'assistant']
    content: Union[List[AnthropicContentBlockType], str]


class AnthropicToolDefinition(BaseModel):
    """Tool definition model."""

    model_config = ConfigDict(extra='allow')

    name: str
    description: str
    input_schema: dict[str, Any] = Field(description='JSON schema object defining expected parameters for the tool')


class AnthropicBuiltInToolUsage(BaseModel):
    """Built-in Tool usages."""

    model_config = ConfigDict(extra='allow')

    type: str
    name: str


class AnthropicThinkingConfig(BaseModel):
    """Thinking configuration model."""

    model_config = ConfigDict(extra='allow')

    budget_tokens: int
    type: Literal['enabled']


class AnthropicMetadata(BaseModel):
    """Request metadata model."""

    model_config = ConfigDict(extra='allow')

    user_id: str


class AnthropicRequest(BaseModel):
    """Main Anthropic API request model."""

    model_config = ConfigDict(extra='allow')

    model: str
    messages: List[AnthropicMessage]
    temperature: Optional[float] = None
    system: list[AnthropicSystemMessage] | str | None = None
    tools: Optional[List[AnthropicToolDefinition|AnthropicBuiltInToolUsage]] = None
    metadata: Optional[AnthropicMetadata] = None
    # max_tokens: int | None = Field(default=32000)
    max_tokens: int | None = None
    thinking: Optional[AnthropicThinkingConfig] = None
    stream: Optional[bool] = None

    def to_dict(self):
        return self.model_dump(mode='json', by_alias=True, exclude_none=True)


class MessageErrorDetail(BaseModel):
    type: str
    message: str


class MessageError(BaseModel):
    type: Literal['error']
    error: MessageErrorDetail
