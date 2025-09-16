"""Provider-neutral request/response exchange primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


@dataclass(slots=True)
class ExchangeRequest:
    """Provider-neutral representation of an API request."""

    channel: str
    model: str
    original_stream: bool
    payload: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    tools: List[Any] = field(default_factory=list)
    extras: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(
        cls,
        payload: Any,
        *,
        channel: str,
        model: str,
        original_stream: bool,
        tools: Optional[Iterable[Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        extras: Optional[Dict[str, Any]] = None,
    ) -> 'ExchangeRequest':
        """Create an exchange request from an arbitrary payload."""

        return cls(
            channel=channel,
            model=model,
            original_stream=original_stream,
            payload=payload,
            metadata=dict(metadata or {}),
            tools=list(tools or []),
            extras=dict(extras or {}),
        )

    def copy_with(self, **updates: Any) -> 'ExchangeRequest':
        """Return a shallow copy of the request with updates applied."""

        data = {
            'channel': self.channel,
            'model': self.model,
            'original_stream': self.original_stream,
            'payload': self.payload,
            'metadata': dict(self.metadata),
            'tools': list(self.tools),
            'extras': dict(self.extras),
        }
        data.update(updates)
        return ExchangeRequest(**data)


@dataclass(slots=True)
class ExchangeResponse:
    """Provider-neutral representation of a completed response."""

    channel: str
    model: str
    payload: Dict[str, Any]
    stream: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExchangeStreamChunk:
    """Represents a chunk of a streamed response."""

    channel: str
    model: str
    event: str
    data: Dict[str, Any]
    finished: bool = False
