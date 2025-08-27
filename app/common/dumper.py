import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import BinaryIO, Dict, Optional

import orjson
from fastapi import Request

from app.common.utils import get_correlation_id
from app.config import ConfigModel


@dataclass
class DumpHandles:
    headers_path: Optional[str]
    request_path: Optional[str]
    response_path: Optional[str]
    response_file: Optional[BinaryIO]
    transformed_request_path: Optional[str]
    transformed_request_file: Optional[BinaryIO]
    pretransformed_response_path: Optional[str]
    pretransformed_response_file: Optional[BinaryIO]
    correlation_id: str


class Dumper:
    def __init__(self, cfg: ConfigModel):
        self.cfg = cfg

    def _ensure_dir(self) -> Optional[str]:
        if not self.cfg.dump_dir:
            return None
        try:
            os.makedirs(self.cfg.dump_dir, exist_ok=True)
            return self.cfg.dump_dir
        except Exception:
            return None

    def _sanitize_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        lower_keys = {k.lower(): k for k in headers.keys()}
        result: Dict[str, str] = {}
        base = {'authorization', 'cookie', 'set-cookie'}
        addl = set(x.lower() for x in (self.cfg.redact_headers or []))
        sensitive = base | addl
        for lk, orig in lower_keys.items():
            if lk in sensitive:
                result[orig] = '***REDACTED***'
            else:
                result[orig] = headers[orig]
        return result

    def begin(self, request: Request, payload: object, correlation_id: Optional[str] = None) -> DumpHandles:
        dump_dir = self._ensure_dir()
        corr_id = correlation_id or get_correlation_id()
        ts = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%S.%fZ')

        headers_path = request_path = response_path = None
        transformed_request_path = pretransformed_response_path = None
        response_file = transformed_request_file = pretransformed_response_file = None

        if dump_dir and self.cfg.dump_headers:
            headers_path = os.path.join(dump_dir, f'{ts}_{corr_id}_1headers.json')
            try:
                hdrs = dict(request.headers)
                sanitized = self._sanitize_headers(hdrs)
                with open(headers_path, 'wb') as f:
                    f.write(orjson.dumps(sanitized, option=orjson.OPT_INDENT_2))
            except Exception:
                headers_path = None

        if dump_dir and self.cfg.dump_requests:
            request_path = os.path.join(dump_dir, f'{ts}_{corr_id}_2request.json')
            try:
                with open(request_path, 'wb') as f:
                    f.write(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
            except Exception:
                request_path = None

        if dump_dir and self.cfg.dump_responses:
            response_path = os.path.join(dump_dir, f'{ts}_{corr_id}_5response.sse')
            try:
                response_file = open(response_path, 'wb')
            except Exception:
                response_file = None

        if dump_dir and self.cfg.dump_transformed_requests:
            transformed_request_path = os.path.join(dump_dir, f'{ts}_{corr_id}_3transformed_request.json')
            try:
                transformed_request_file = open(transformed_request_path, 'wb')
            except Exception:
                transformed_request_file = None

        if dump_dir and self.cfg.dump_pretransformed_responses:
            pretransformed_response_path = os.path.join(dump_dir, f'{ts}_{corr_id}_4pretransformed_response.sse')
            try:
                pretransformed_response_file = open(pretransformed_response_path, 'wb')
            except Exception:
                pretransformed_response_file = None

        return DumpHandles(
            headers_path=headers_path,
            request_path=request_path,
            response_path=response_path,
            response_file=response_file,
            transformed_request_path=transformed_request_path,
            transformed_request_file=transformed_request_file,
            pretransformed_response_path=pretransformed_response_path,
            pretransformed_response_file=pretransformed_response_file,
            correlation_id=corr_id,
        )

    def write_response_chunk(self, handles: DumpHandles, chunk: bytes | str) -> None:
        f = handles.response_file
        if not f or not chunk:
            return
        try:
            if isinstance(chunk, str):
                chunk = bytes(chunk, encoding='utf-8')
            f.write(chunk)
            f.flush()
        except Exception as e:
            print(f'exception dumping chunk: {e}')

    def close(self, handles: DumpHandles) -> None:
        # Close response file
        if handles.response_file:
            try:
                handles.response_file.close()
            except Exception:
                pass

        # Close transformed request file
        if handles.transformed_request_file:
            try:
                handles.transformed_request_file.close()
            except Exception:
                pass

        # Close pretransformed response file
        if handles.pretransformed_response_file:
            try:
                handles.pretransformed_response_file.close()
            except Exception:
                pass

    def write_transformed_request(self, handles: DumpHandles, transformed_request: dict) -> None:
        """Write the transformed request after request transformers are applied."""
        f = handles.transformed_request_file
        if not f:
            return

        try:
            f.write(orjson.dumps(transformed_request, option=orjson.OPT_INDENT_2))
            f.flush()
        except Exception:
            pass

    def write_pretransformed_response(self, handles: DumpHandles, chunk: bytes | str) -> None:
        """Write raw response chunks before response transformers are applied."""
        f = handles.pretransformed_response_file
        if not f or not chunk:
            return
        try:
            if isinstance(chunk, str):
                chunk = chunk.encode('utf-8')
            f.write(chunk)
            f.flush()
        except Exception:
            pass
