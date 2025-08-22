import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import BinaryIO, Dict, Optional

from fastapi import Request

from app.config import ConfigModel


@dataclass
class DumpHandles:
    headers_path: Optional[str]
    request_path: Optional[str]
    response_path: Optional[str]
    response_file: Optional[BinaryIO]


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
        corr_id = correlation_id or uuid.uuid4().hex
        ts = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%S.%fZ')

        headers_path = request_path = response_path = None
        response_file: Optional[BinaryIO] = None

        if dump_dir and self.cfg.dump_headers:
            headers_path = os.path.join(dump_dir, f'{ts}_{corr_id}_headers.json')
            try:
                hdrs = dict(request.headers)
                sanitized = self._sanitize_headers(hdrs)
                with open(headers_path, 'w', encoding='utf-8') as f:
                    json.dump(sanitized, f, ensure_ascii=False, indent=2)
            except Exception:
                headers_path = None

        if dump_dir and self.cfg.dump_requests:
            request_path = os.path.join(dump_dir, f'{ts}_{corr_id}_request.json')
            try:
                with open(request_path, 'w', encoding='utf-8') as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
            except Exception:
                request_path = None

        if dump_dir and self.cfg.dump_responses:
            response_path = os.path.join(dump_dir, f'{ts}_{corr_id}_response.sse')
            try:
                response_file = open(response_path, 'wb')
            except Exception:
                response_file = None

        return DumpHandles(
            headers_path=headers_path,
            request_path=request_path,
            response_path=response_path,
            response_file=response_file,
        )

    def write_chunk(self, handles: DumpHandles, chunk: bytes) -> None:
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
        f = handles.response_file
        if not f:
            return
        try:
            f.close()
        except Exception:
            pass
