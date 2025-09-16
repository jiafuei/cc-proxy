"""Provider descriptor definitions describing default behaviour per backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping

from app.providers.types import ChannelName, ProviderType

TransformerConfig = List[Dict[str, object]]
ChannelTransformerMap = Mapping[ChannelName, Mapping[str, TransformerConfig]]


@dataclass(frozen=True)
class ProviderDescriptor:
    """Describes capabilities and defaults for a provider backend."""

    type: ProviderType
    base_url_suffixes: Mapping[str, str]
    default_transformers: ChannelTransformerMap
    supports_streaming: bool
    supports_count_tokens: bool
    supports_responses: bool

    @property
    def operations(self) -> List[str]:
        """Return the supported operation keys."""

        return list(self.base_url_suffixes.keys())
