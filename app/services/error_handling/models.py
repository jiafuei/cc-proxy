"""Error models for API responses."""

from typing import Optional

from pydantic import BaseModel, Field


class ClaudeErrorDetail(BaseModel):
    """Claude error detail model."""

    type: str
    message: str


class ClaudeError(BaseModel):
    """Claude error response model."""

    type: str = Field(default='error')
    error: ClaudeErrorDetail
    request_id: Optional[str] = None

    def to_dict(self):
        return self.model_dump(mode='json', exclude_none=True)
