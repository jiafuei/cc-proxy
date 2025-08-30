"""Request body transformer using JSONPath for flexible modifications."""

from typing import Any, Dict, Tuple
import copy

from jsonpath_ng import parse

from app.services.transformers.interfaces import RequestTransformer


class RequestBodyTransformer(RequestTransformer):
    """Transformer that modifies request body content using JSONPath expressions.

    Parameters
    - key: path expression (JSONPath when jsonPath=True, otherwise treated as JSONPath too)
    - value: value to use for operation
    - op: one of 'set', 'delete', 'append', 'prepend', 'merge'
    - jsonPath: boolean flag; kept for API compatibility (always uses JSONPath)
    """

    def __init__(self, logger, key: str = "", value: Any = None, op: str = "set", jsonPath: bool = True):
        super().__init__(logger)
        self.key = key
        self.value = value
        self.op = op.lower()
        self.jsonPath = bool(jsonPath)

        valid_ops = {"set", "delete", "append", "prepend", "merge"}
        if self.op not in valid_ops:
            raise ValueError(f"Invalid operation '{self.op}'. Must be one of: {valid_ops}")

        if not self.key:
            raise ValueError("'key' parameter is required and must be a JSONPath expression")

        # Pre-compile JSONPath
        try:
            self.expr = parse(self.key)
        except Exception as e:
            raise ValueError(f"Invalid JSONPath expression '{self.key}': {e}")

    async def transform(self, params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
        request = params["request"]
        headers = params["headers"]

        transformed_request = copy.deepcopy(request)

        try:
            matches = list(self.expr.find(transformed_request))

            if self.op == "delete":
                for match in matches:
                    self._delete_match(transformed_request, match)

            elif self.op == "set":
                for match in matches:
                    self._set_match(transformed_request, match, self.value)

            elif self.op in {"append", "prepend"}:
                for match in matches:
                    self._list_insert_match(transformed_request, match, self.value, self.op)

            elif self.op == "merge":
                for match in matches:
                    self._merge_match(transformed_request, match, self.value)

            self.logger.debug(f"Applied {self.op} operation using JSONPath '{self.key}'")

        except Exception as e:
            self.logger.error(f"Failed to apply {self.op} operation using JSONPath '{self.key}': {e}")
            return request, headers

        return transformed_request, headers

    def _delete_match(self, data: Dict[str, Any], match) -> None:
        context = match.context.value
        path = match.path
        try:
            if hasattr(path, "index"):
                # list index
                idx = path.index
                if isinstance(context, list) and 0 <= idx < len(context):
                    context.pop(idx)
            elif hasattr(path, "fields"):
                # field access
                for f in path.fields:
                    if isinstance(context, dict) and f in context:
                        del context[f]
        except Exception:
            raise

    def _set_match(self, data: Dict[str, Any], match, value: Any) -> None:
        context = match.context.value
        path = match.path
        try:
            if hasattr(path, "index"):
                idx = path.index
                if isinstance(context, list):
                    while len(context) <= idx:
                        context.append(None)
                    context[idx] = value
            elif hasattr(path, "fields"):
                # set fields on dict
                for f in path.fields:
                    if isinstance(context, dict):
                        context[f] = value
        except Exception:
            raise

    def _list_insert_match(self, data: Dict[str, Any], match, value: Any, op: str) -> None:
        context = match.value
        # match.value is the matched object itself
        if isinstance(context, list):
            if op == "append":
                context.append(value)
            else:
                context.insert(0, value)
        else:
            raise ValueError("Target for append/prepend is not a list")

    def _merge_match(self, data: Dict[str, Any], match, value: Any) -> None:
        context = match.value
        if not isinstance(context, dict) or not isinstance(value, dict):
            raise ValueError("Merge requires dict target and dict value")
        context.update(value)