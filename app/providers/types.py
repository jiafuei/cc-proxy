"""Provider-related enumerations and typing helpers."""

from __future__ import annotations

from enum import Enum
from typing import Literal


class ProviderType(str, Enum):
    """Supported provider backend identifiers."""

    ANTHROPIC = 'anthropic'
    OPENAI = 'openai'
    OPENAI_RESPONSES = 'openai-responses'
    GEMINI = 'gemini'


ChannelName = Literal['claude', 'codex']


def all_channels() -> tuple[ChannelName, ChannelName]:
    """Return the supported channel names."""

    return ('claude', 'codex')
