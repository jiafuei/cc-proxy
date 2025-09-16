"""Request/response dumping helpers for debugging and observability."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, BinaryIO, Dict, List, Optional

import orjson
from fastapi import Request

from app.config.models import ConfigModel
from app.context import get_correlation_id


class HeaderSanitizer:
    """Redact sensitive headers before writing them to disk."""

    def __init__(self, redact_headers: Optional[List[str]] = None):
        base_sensitive = {'authorization', 'cookie', 'set-cookie'}
        additional = {value.lower() for value in (redact_headers or [])}
        self.sensitive_headers = base_sensitive | additional

    def sanitize(self, headers: Dict[str, str]) -> Dict[str, str]:
        result: Dict[str, str] = {}
        lower_keys = {key.lower(): key for key in headers.keys()}

        for lower_key, original_key in lower_keys.items():
            if lower_key in self.sensitive_headers:
                result[original_key] = '***REDACTED***'
            else:
                result[original_key] = headers[original_key]
        return result


class DumpType(Enum):
    """Types of artifacts captured by the dumper."""

    ORIGINAL_HEADERS = 'original_headers'
    TRANSFORMED_HEADERS = 'transformed_headers'
    ORIGINAL_REQUEST = 'original_request'
    TRANSFORMED_REQUEST = 'transformed_request'
    PRETRANSFORMED_RESPONSE = 'pretransformed_response'
    FINAL_RESPONSE = 'final_response'


class DumpPathGenerator:
    """Derive deterministic filenames for dump artifacts."""

    EXTENSIONS = {
        DumpType.ORIGINAL_HEADERS: '.json',
        DumpType.TRANSFORMED_HEADERS: '.json',
        DumpType.ORIGINAL_REQUEST: '.json',
        DumpType.TRANSFORMED_REQUEST: '.json',
        DumpType.PRETRANSFORMED_RESPONSE: '.sse',
        DumpType.FINAL_RESPONSE: '.sse',
    }

    ORDERING = {
        DumpType.ORIGINAL_HEADERS: 1,
        DumpType.TRANSFORMED_HEADERS: 2,
        DumpType.ORIGINAL_REQUEST: 3,
        DumpType.TRANSFORMED_REQUEST: 4,
        DumpType.PRETRANSFORMED_RESPONSE: 5,
        DumpType.FINAL_RESPONSE: 6,
    }

    def generate_path(self, base_path: str, dump_type: DumpType) -> str:
        number = self.ORDERING[dump_type]
        extension = self.EXTENSIONS[dump_type]
        return f'{base_path}_{number}_{dump_type.value}{extension}'


@dataclass
class DumpFiles:
    """File metadata tracked during a request lifecycle."""

    original_headers: Optional[str] = None
    transformed_headers: Optional[str] = None
    original_request: Optional[str] = None
    transformed_request: Optional[str] = None
    pretransformed_response: Optional[str] = None
    pretransformed_response_file: Optional[BinaryIO] = None
    final_response: Optional[str] = None
    final_response_file: Optional[BinaryIO] = None


@dataclass
class DumpHandles:
    """Return object handed to router/provider layers for streaming dumps."""

    files: DumpFiles
    correlation_id: str
    base_path: str


class Dumper:
    """Persist sanitized request/response data for debugging."""

    def __init__(self, cfg: ConfigModel):
        self.cfg = cfg
        self.sanitizer = HeaderSanitizer(cfg.redact_headers)
        self.path_generator = DumpPathGenerator()

    def _ensure_dir(self) -> Optional[str]:
        if not self.cfg.dump_dir:
            return None
        try:
            os.makedirs(self.cfg.dump_dir, exist_ok=True)
            return self.cfg.dump_dir
        except Exception:
            return None

    def _sanitize_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        return self.sanitizer.sanitize(headers)

    def _get_file_path(self, base_path: str, dump_type: DumpType) -> str:
        return self.path_generator.generate_path(base_path, dump_type)

    def _write_json_file(self, file_path: str, data: Any) -> bool:
        try:
            with open(file_path, 'wb') as handle:
                handle.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))
            return True
        except Exception:
            return False

    def _open_streaming_file(self, file_path: str) -> Optional[BinaryIO]:
        try:
            return open(file_path, 'wb')
        except Exception:
            return None

    def begin(self, request: Request, payload: object, correlation_id: Optional[str] = None) -> DumpHandles:
        dump_dir = self._ensure_dir()
        corr_id = correlation_id or get_correlation_id()
        timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%S.%fZ')

        files = DumpFiles()
        base_path = os.path.join(dump_dir, f'{timestamp}_{corr_id}') if dump_dir else ''
        handles = DumpHandles(files=files, correlation_id=corr_id, base_path=base_path)

        if not dump_dir:
            return handles

        if self.cfg.dump_headers:
            headers = dict(request.headers)
            sanitized = self._sanitize_headers(headers)
            self._write_data(handles, DumpType.ORIGINAL_HEADERS, sanitized)

        if self.cfg.dump_requests:
            self._write_data(handles, DumpType.ORIGINAL_REQUEST, payload)

        if self.cfg.dump_responses:
            path = self._get_file_path(base_path, DumpType.PRETRANSFORMED_RESPONSE)
            files.pretransformed_response = path
            files.pretransformed_response_file = self._open_streaming_file(path)

            path = self._get_file_path(base_path, DumpType.FINAL_RESPONSE)
            files.final_response = path
            files.final_response_file = self._open_streaming_file(path)

        return handles

    def _should_dump(self, dump_type: DumpType) -> bool:
        return {
            DumpType.ORIGINAL_HEADERS: self.cfg.dump_headers,
            DumpType.TRANSFORMED_HEADERS: self.cfg.dump_headers,
            DumpType.ORIGINAL_REQUEST: self.cfg.dump_requests,
            DumpType.TRANSFORMED_REQUEST: self.cfg.dump_requests,
            DumpType.PRETRANSFORMED_RESPONSE: self.cfg.dump_responses,
            DumpType.FINAL_RESPONSE: self.cfg.dump_responses,
        }.get(dump_type, False)

    def _write_data(self, handles: DumpHandles, dump_type: DumpType, data: Any) -> bool:
        if not handles.base_path or not self._should_dump(dump_type):
            return False

        file_path = self._get_file_path(handles.base_path, dump_type)
        return self._write_json_file(file_path, data)

    def _write_streaming_data(self, handle: Optional[BinaryIO], data: bytes | str) -> bool:
        if not handle or not data:
            return False

        try:
            if isinstance(data, str):
                data = data.encode('utf-8')
            handle.write(data)
            handle.flush()
            return True
        except Exception:
            return False

    def write_transformed_headers(self, handles: DumpHandles, headers: Dict[str, str]) -> None:
        sanitized = self._sanitize_headers(headers)
        self._write_data(handles, DumpType.TRANSFORMED_HEADERS, sanitized)

    def write_transformed_request(self, handles: DumpHandles, request: Dict[str, Any]) -> None:
        self._write_data(handles, DumpType.TRANSFORMED_REQUEST, request)

    def write_pretransformed_response(self, handles: DumpHandles, chunk: bytes | str) -> None:
        self._write_streaming_data(handles.files.pretransformed_response_file, chunk)

    def write_response_chunk(self, handles: DumpHandles, chunk: bytes | str) -> None:
        self._write_streaming_data(handles.files.final_response_file, chunk)

    def close(self, handles: DumpHandles) -> None:
        for file_handle in [handles.files.pretransformed_response_file, handles.files.final_response_file]:
            if file_handle:
                try:
                    file_handle.close()
                except Exception:
                    pass


__all__ = ['DumpHandles', 'DumpFiles', 'DumpPathGenerator', 'DumpType', 'Dumper', 'HeaderSanitizer']
