"""Shared retry wrapper for BAML LLM calls.

Backoff is conditional on the error class, not a flat wait for everything:

- Parse/validation and finish-reason (truncation) errors retry almost
  immediately — a wait cannot fix a malformed or truncated generation, only a
  fresh generation can.
- Transient transport errors (HTTP 429/5xx, timeouts, connection errors) keep a
  real backoff.
- Non-retryable errors (HTTP 401/4xx auth/bad-request, invalid argument, abort)
  fail fast — retrying cannot succeed, so we stop immediately.
"""

import re
import time
from typing import Callable, TypeVar

import mlflow
from baml_py.errors import (
    BamlAbortError,
    BamlClientError,
    BamlClientFinishReasonError,
    BamlClientHttpError,
    BamlInvalidArgumentError,
    BamlTimeoutError,
    BamlValidationError,
)
from loguru import logger

from src.schemas.agent_error import AgentError

T = TypeVar("T")

DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY_S = 30      # transient transport errors (429/5xx/timeouts)
DEFAULT_FAST_RETRY_DELAY_S = 1  # parse/validation/finish-reason errors

# Retry classification.
_RETRY_NOW = "retry_now"        # regenerate immediately; a wait won't help
_RETRY_BACKOFF = "retry_backoff"  # transient; back off then retry
_FAIL_FAST = "fail_fast"        # cannot succeed on retry; stop now


def _http_status(e: Exception) -> int | None:
    """Best-effort HTTP status code from a BAML client error.

    Prefer the structured attribute (always set on BamlClientHttpError). The
    string fallback anchors on a `status_code=`/`code=` label only — no bare
    3-digit match, which could pick up an unrelated number (token count, id,
    line). If neither is found we return None, and the caller backs off rather
    than guessing a status.
    """
    code = getattr(e, "status_code", None)
    if isinstance(code, int):
        return code
    m = re.search(r"(?:status_code|code)=(\d{3})", str(e))
    return int(m.group(1)) if m else None


def _classify(e: Exception) -> tuple[str, str]:
    """Map an exception to (action, human-readable reason)."""
    # Output unusable for a local reason — only a fresh generation can fix it.
    if isinstance(e, (BamlValidationError, BamlClientFinishReasonError)):
        return _RETRY_NOW, f"{type(e).__name__} (regenerate; waiting won't help)"

    # HTTP errors: retry transient statuses, fail fast on the rest.
    # NOTE: check before BamlClientError — BamlClientHttpError subclasses it.
    if isinstance(e, BamlClientHttpError):
        code = _http_status(e)
        if code == 429 or (code is not None and code >= 500) or code == 408:
            return _RETRY_BACKOFF, f"HTTP {code} (transient, back off)"
        if code is None:
            return _RETRY_BACKOFF, "HTTP error, unknown status (back off)"
        return _FAIL_FAST, f"HTTP {code} (non-retryable)"

    # Timeouts and other client/network errors are transient.
    if isinstance(e, (BamlTimeoutError, BamlClientError)):
        return _RETRY_BACKOFF, f"{type(e).__name__} (transient, back off)"

    # Programming errors / cancellation never succeed on retry.
    if isinstance(e, (BamlInvalidArgumentError, BamlAbortError)):
        return _FAIL_FAST, f"{type(e).__name__} (non-retryable)"

    # Unknown error: be conservative and back off.
    return _RETRY_BACKOFF, f"{type(e).__name__} (unclassified, back off)"


def call_baml_with_retry(
    fn: Callable[[], T],
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay_s: float = DEFAULT_RETRY_DELAY_S,
    fast_retry_delay_s: float = DEFAULT_FAST_RETRY_DELAY_S,
    context_name: str = "baml_call",
) -> T | AgentError:
    """
    Call a BAML function with conditional retry logic and MLflow error logging.

    Backoff depends on the error class (see module docstring): parse/finish-reason
    errors retry after ``fast_retry_delay_s``; transient transport errors after
    ``retry_delay_s``; non-retryable errors stop immediately.

    Returns the BAML result on success, or AgentError once retries are exhausted
    or a non-retryable error occurs.
    """
    last_exception: Exception | None = None
    attempts = 0

    for attempt in range(1, max_retries + 1):
        attempts = attempt
        try:
            return fn()
        except Exception as e:
            last_exception = e
            error_type = type(e).__name__
            raw_output = getattr(e, "raw_output", None)
            action, reason = _classify(e)

            logger.warning(
                f"[{context_name}] Attempt {attempt}/{max_retries} failed "
                f"({error_type}; {reason}): {str(e)[:200]}"
            )
            if raw_output:
                logger.debug(
                    f"[{context_name}] Raw LLM output ({len(raw_output)} chars): "
                    f"{raw_output[:500]}"
                )

            # Log to MLflow (best-effort)
            try:
                with mlflow.start_span(
                    name=f"{context_name}_retry_{attempt}",
                    span_type="CHAIN",
                ) as retry_span:
                    retry_span.set_attributes(
                        {
                            "error_type": error_type,
                            "retry_action": action,
                            "attempt": attempt,
                            "max_retries": max_retries,
                        }
                    )
                    retry_span.set_inputs({"error_message": str(e)[:2000]})
                    if raw_output:
                        retry_span.set_outputs(
                            {"raw_output": str(raw_output)[:2000]}
                        )
                    retry_span.set_status("ERROR")
            except Exception:
                pass

            # Non-retryable: stop now, don't burn the remaining attempts.
            if action == _FAIL_FAST:
                logger.error(
                    f"[{context_name}] Non-retryable error ({error_type}); failing fast."
                )
                break

            if attempt < max_retries:
                delay = fast_retry_delay_s if action == _RETRY_NOW else retry_delay_s
                logger.info(
                    f"[{context_name}] Waiting {delay}s before retry ({reason})..."
                )
                time.sleep(delay)

    # Retries exhausted or failed fast - return AgentError (don't raise)
    assert last_exception is not None
    logger.error(
        f"[{context_name}] Giving up after {attempts} attempt(s). "
        f"Last error: {type(last_exception).__name__}"
    )
    return AgentError(
        error_type=type(last_exception).__name__,
        error_message=str(last_exception)[:2000],
        context_name=context_name,
        raw_output=str(getattr(last_exception, "raw_output", None))[:2000]
        if getattr(last_exception, "raw_output", None)
        else None,
        attempts=attempts,
    )
