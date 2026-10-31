"""MiniMax client.

MiniMax exposes an Anthropic-compatible API at ``https://api.minimax.io/anthropic/v1``
and an OpenAI-compatible one at ``https://api.minimax.io/v1``. Both are
supported; the Anthropic route is the default because MiniMax's own quickstart
uses it and it accepts a top-level ``system`` prompt.

The client is deliberately thin: it sends one chat request and returns the
text. Everything else - chunking, retries across passes, scoring - lives in
:mod:`humanizer.pipeline`.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

import httpx

from .config import Config


class MiniMaxError(RuntimeError):
    """Raised when the API returns an error we cannot recover from."""


class MiniMaxClient:
    """A minimal chat client for MiniMax.

    Parameters
    ----------
    config:
        The resolved configuration. ``config.api_key`` must be set.
    """

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        if not self.config.has_key:
            raise MiniMaxError(self.config.missing_key_message())

    # ------------------------------------------------------------------
    # request building
    # ------------------------------------------------------------------
    def _endpoint(self) -> str:
        base = self.config.base_url.rstrip("/")
        if self.config.api_style == "anthropic":
            if base.endswith("/anthropic/v1"):
                return f"{base}/messages"
            if base.endswith("/v1"):
                return f"{base}/messages"
            return f"{base}/anthropic/v1/messages"
        if base.endswith("/anthropic/v1"):
            return f"{base}/chat/completions"
        if base.endswith("/v1"):
            return f"{base}/chat/completions"
        return f"{base}/v1/chat/completions"

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }
        if self.config.api_style == "anthropic":
            headers["x-api-key"] = self.config.api_key
            headers["anthropic-version"] = "2023-06-01"
        return headers

    def _payload(
        self,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        if self.config.api_style == "anthropic":
            return {
                "model": self.config.model,
                "max_tokens": max_tokens,
                "system": system,
                "temperature": temperature,
                "messages": [{"role": "user", "content": user}],
            }
        return {
            "model": self.config.model,
            # max_tokens is deprecated on the OpenAI-compatible endpoint;
            # max_completion_tokens is the current name.
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
            "top_p": self.config.top_p,
            # reasoning_split keeps the model's internal reasoning in a
            # separate field instead of inline in the content.
            "reasoning_split": True,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }

    # ------------------------------------------------------------------
    # response parsing
    # ------------------------------------------------------------------
    #: MiniMax's reasoning models wrap their answer in <think>...</think>.
    #: Those blocks are internal reasoning and must not reach the output.
    _THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
    _THINK_OPEN = re.compile(r"^\s*<think>.*", re.DOTALL | re.IGNORECASE)

    @classmethod
    def _strip_thinking(cls, text: str) -> str:
        """Remove the model's reasoning blocks from its answer."""
        if not text:
            return text
        cleaned = cls._THINK_BLOCK.sub("", text)
        # An unterminated block runs to the end of the reply.
        cleaned = cls._THINK_OPEN.sub("", cleaned)
        return cleaned.strip()

    @classmethod
    def _extract_text(cls, payload: dict[str, Any]) -> str:
        """Pull the assistant text out of either response shape."""
        # Anthropic shape
        content = payload.get("content")
        if isinstance(content, list):
            parts = [
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            if parts:
                return cls._strip_thinking("".join(parts))
            # Some responses return a thinking block plus a text block.
            for block in content:
                if isinstance(block, dict) and block.get("type") == "thinking":
                    continue
            # Fall back to any string content.
            strings = [
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("text")
            ]
            if strings:
                return cls._strip_thinking("".join(strings))

        # OpenAI shape
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message", {})
            text = message.get("content")
            if isinstance(text, str):
                return cls._strip_thinking(text)
            if isinstance(text, list):
                return cls._strip_thinking(
                    "".join(
                        part.get("text", "") for part in text if isinstance(part, dict)
                    )
                )
            # NOTE: with reasoning_split enabled, ``reasoning_content`` holds
            # the model's INTERNAL REASONING, never the answer. Returning it
            # here put the model's scratchpad into the user's document - the
            # rewritten output was the model's own reasoning about the task.
            # An empty content field means the model did not produce an
            # answer; the caller retries. Never substitute reasoning.
            #
            # A response whose content is empty but whose reasoning is
            # present is a failure, not an answer.

        # Some MiniMax responses use a flat "reply" field.
        for key in ("reply", "output", "text"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return cls._strip_thinking(value)

        return ""

    # ------------------------------------------------------------------
    # the call
    # ------------------------------------------------------------------
    def chat(
        self,
        user: str,
        system: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Send one chat request and return the assistant's text.

        Retries on transient failures (429, 5xx, network errors) with an
        exponential backoff, up to ``config.max_retries`` attempts.
        """
        temperature = self.config.temperature if temperature is None else temperature
        max_tokens = self.config.max_tokens if max_tokens is None else max_tokens
        url = self._endpoint()
        payload = self._payload(system, user, temperature, max_tokens)

        last_error: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                with httpx.Client(timeout=self.config.timeout) as client:
                    response = client.post(url, headers=self._headers(), json=payload)
                if response.status_code >= 400:
                    body = response.text[:600]
                    if response.status_code in (429, 500, 502, 503, 504):
                        last_error = MiniMaxError(
                            f"MiniMax returned {response.status_code}: {body}"
                        )
                        delay = 2**attempt
                        time.sleep(delay)
                        continue
                    raise MiniMaxError(
                        f"MiniMax returned {response.status_code}: {body}"
                    )
                data = response.json()
                text = self._extract_text(data)
                if not text:
                    last_error = MiniMaxError(
                        f"MiniMax returned no text: {json.dumps(data)[:600]}"
                    )
                    time.sleep(2**attempt)
                    continue
                return text
            except httpx.HTTPError as exc:
                last_error = exc
                time.sleep(2**attempt)

        raise MiniMaxError(f"MiniMax request failed after {self.config.max_retries} "
                            f"attempts: {last_error}")

    def list_models(self) -> list[str]:
        """Return the model ids this key can use (best effort)."""
        base = self.config.base_url.rstrip("/")
        url = base.replace("/anthropic/v1", "").replace("/v1", "") + "/v1/models"
        try:
            with httpx.Client(timeout=self.config.timeout) as client:
                response = client.get(url, headers=self._headers())
            if response.status_code >= 400:
                return []
            payload = response.json()
            entries = payload.get("data", [])
            return [e.get("id", "") for e in entries if isinstance(e, dict)]
        except httpx.HTTPError:
            return []

    def check(self) -> tuple[bool, str]:
        """Verify the key works. Returns ``(ok, message)``."""
        try:
            text = self.chat(
                user="Reply with the single word: ready",
                system="You reply with one word only.",
                max_tokens=16,
            )
        except MiniMaxError as exc:
            return False, str(exc)
        return True, f"Connected to {self.config.model}: {text[:80]}"


class EchoBackend:
    """A stub backend used by the tests and by ``--dry-run``.

    It returns a marker instead of calling the API, so the pipeline can be
    exercised end to end without a key or a network.
    """

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()

    def chat(
        self,
        user: str,
        system: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        return f"[dry-run] {user.strip()[:200]}"

    def list_models(self) -> list[str]:
        return ["dry-run"]

    def check(self) -> tuple[bool, str]:
        return True, "Dry-run backend: no API calls will be made."


def build_client(config: Config | None = None, dry_run: bool = False):
    """Return a client for the configured backend."""
    config = config or Config()
    if dry_run:
        return EchoBackend(config)
    return MiniMaxClient(config)
