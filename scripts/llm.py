"""Unified client interface for LLM inference.

A thin wrapper that homogenizes calls across Ollama (local) and the
proprietary APIs. Clients return an ``LLMResponse`` with text +
optional thinking trace + raw provider data.

Currently implemented: ``OllamaClient``, ``GeminiClient``.
To be added: ``OpenAIClient`` (GPT-5.x), ``AnthropicClient`` (Claude).
``get_client(model)`` is the factory that picks a backend by model
name prefix.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import httpx
from dotenv import load_dotenv


class AllKeysExhausted(Exception):
    """Raised by ``GeminiClient`` when every key in the pool has
    returned a non-retriable status (429 / 401 / 403). The caller can
    catch this to abort early and save partial results."""


@dataclass
class LLMResponse:
    text: str                                  # main output (JSON string for cloze)
    thinking: str | None                       # reasoning trace if exposed
    model: str                                 # model id used
    reasoning_enabled: bool                    # whether thinking was requested
    raw: dict[str, Any] = field(default_factory=dict)


class LLMClient(Protocol):
    """Contract every backend implements."""

    @property
    def model(self) -> str: ...

    def complete(
        self,
        prompt: str,
        *,
        reasoning: bool = False,
        json_mode: bool = True,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse: ...


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

class OllamaClient:
    """Ollama ``/api/chat`` client.

    For Qwen3 / DeepSeek-R1 / GPT-OSS and other hybrid-thinking models,
    pass ``reasoning=True`` to engage thinking. The reasoning trace is
    returned in ``LLMResponse.thinking`` when the runtime exposes it.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout: float = 300.0,
    ) -> None:
        self._model = model
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout)

    @property
    def model(self) -> str:
        return self._model

    def complete(
        self,
        prompt: str,
        *,
        reasoning: bool = False,
        json_mode: bool = True,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        body: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": bool(reasoning),
            "options": {"temperature": temperature},
        }
        if max_tokens is not None:
            body["options"]["num_predict"] = max_tokens
        if json_mode:
            body["format"] = "json"

        r = self._client.post(f"{self.base_url}/api/chat", json=body)
        r.raise_for_status()
        data = r.json()
        msg = data.get("message") or {}
        return LLMResponse(
            text=msg.get("content") or "",
            thinking=msg.get("thinking") or None,
            model=self._model,
            reasoning_enabled=bool(reasoning),
            raw=data,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "OllamaClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Gemini (Google AI Studio)
# ---------------------------------------------------------------------------

class GeminiClient:
    """Google AI Studio (Gemini) REST client with multi-key rotation.

    Uses the v1beta ``generateContent`` endpoint directly. Reasoning is
    controlled via ``thinkingConfig.thinkingBudget`` — 0 disables, -1
    enables dynamic thinking budget. JSON mode is requested via
    ``responseMimeType: "application/json"``.

    Reads API keys from (in priority order): ``GEMINI_API_KEY``,
    ``GEMINI_API_KEY_2``, ``GEMINI_API_KEY_GCLOUD``,
    ``GEMINI_API_KEY_FREE``, ``GOOGLE_API_KEY``. All keys that are set
    are pooled. On 429 the offending key is marked exhausted (for the
    rest of this Python process) and the client rotates to the next
    available key. Transient 5xx errors retry the same key with
    exponential backoff.
    """

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        api_keys: list[tuple[str, str]] | None = None,
        timeout: float = 300.0,
    ) -> None:
        self._model = model
        if api_keys is not None:
            self._keys = list(api_keys)
        elif api_key is not None:
            self._keys = [("explicit", api_key)]
        else:
            self._keys = _resolve_gemini_keys()
        if not self._keys:
            raise SystemExit(
                "no Gemini API key found. set one or more of "
                "GEMINI_API_KEY, GEMINI_API_KEY_2, GEMINI_API_KEY_GCLOUD, "
                "GEMINI_API_KEY_FREE, or GOOGLE_API_KEY in env or .env"
            )
        self._key_idx = 0  # current preferred index
        self._exhausted: set[int] = set()
        self._switch_logged: set[int] = set()
        self._client = httpx.Client(timeout=timeout)

    @property
    def model(self) -> str:
        return self._model

    @property
    def key_pool(self) -> list[str]:
        """Names of available keys (values not exposed)."""
        return [name for name, _ in self._keys]

    def complete(
        self,
        prompt: str,
        *,
        reasoning: bool = False,
        json_mode: bool = True,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        url = f"{self.BASE_URL}/models/{self._model}:generateContent"
        gen_cfg: dict[str, Any] = {
            "temperature": temperature,
            "thinkingConfig": {
                "thinkingBudget": -1 if reasoning else 0,
            },
        }
        if max_tokens is not None:
            gen_cfg["maxOutputTokens"] = max_tokens
        if json_mode:
            gen_cfg["responseMimeType"] = "application/json"

        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": gen_cfg,
        }

        data = self._post_with_retry(url, body)

        text_parts: list[str] = []
        thinking_parts: list[str] = []
        for cand in (data.get("candidates") or []):
            content = cand.get("content") or {}
            for p in (content.get("parts") or []):
                t = p.get("text") or ""
                if p.get("thought"):
                    thinking_parts.append(t)
                else:
                    text_parts.append(t)
            break  # only first candidate

        return LLMResponse(
            text="".join(text_parts),
            thinking="".join(thinking_parts) or None,
            model=self._model,
            reasoning_enabled=bool(reasoning),
            raw=data,
        )

    def _next_active_key(self) -> tuple[int, str, str] | None:
        """Find the next key not yet exhausted, advancing _key_idx."""
        for offset in range(len(self._keys)):
            idx = (self._key_idx + offset) % len(self._keys)
            if idx not in self._exhausted:
                self._key_idx = idx
                name, value = self._keys[idx]
                return idx, name, value
        return None

    # Status codes that take a key out of the pool for the rest of this
    # session: rate-limit (429), auth/permission failures (401/403), and
    # quota errors (some Google variants of 402/forbidden).
    _ROTATE_STATUSES = (401, 403, 429)

    def _post_with_retry(self, url: str, body: dict) -> dict:
        """Try keys in priority order. 401/403/429 → mark exhausted +
        rotate. 5xx → retry same key with exponential backoff (up to 2
        retries). Header auth used so keys never appear in URLs/logs."""
        last_status: int | None = None
        last_text: str = ""
        for _ in range(len(self._keys)):
            picked = self._next_active_key()
            if picked is None:
                break
            idx, name, key = picked
            headers = {"x-goog-api-key": key}
            backoff = 2.0
            transient_attempts = 0
            while True:
                r = self._client.post(url, json=body, headers=headers)
                last_status = r.status_code
                last_text = r.text[:200]
                if r.status_code == 200:
                    return r.json()
                if r.status_code in self._ROTATE_STATUSES:
                    self._exhausted.add(idx)
                    if idx not in self._switch_logged:
                        self._switch_logged.add(idx)
                        remaining = [
                            n for j, (n, _) in enumerate(self._keys)
                            if j not in self._exhausted
                        ]
                        kind = {
                            401: "401 Unauthorized",
                            403: "403 Forbidden",
                            429: "429 Too Many Requests",
                        }.get(r.status_code, f"{r.status_code}")
                        sys.stderr.write(
                            f"[gemini] key {name!r} returned {kind}; "
                            f"marking exhausted. remaining keys: "
                            f"{remaining}\n"
                        )
                        sys.stderr.flush()
                    self._key_idx = (idx + 1) % len(self._keys)
                    break  # advance to next key
                if (r.status_code in (500, 502, 503, 504)
                        and transient_attempts < 2):
                    transient_attempts += 1
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 30.0)
                    continue
                # Non-retriable error on this key
                r.raise_for_status()
                break  # unreachable but keeps mypy quiet
        raise AllKeysExhausted(
            f"all {len(self._keys)} Gemini keys exhausted "
            f"(last status: {last_status}, body: {last_text!r})"
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "GeminiClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


_GEMINI_KEY_VARS = (
    "GEMINI_API_KEY",
    "GEMINI_API_KEY_2",
    "GEMINI_API_KEY_3",
    "GEMINI_API_KEY_GCLOUD",
    "GEMINI_API_KEY_FREE",
    "GOOGLE_API_KEY",
)


def _resolve_gemini_keys() -> list[tuple[str, str]]:
    """Return all set Gemini keys as ``(name, value)`` in priority order."""
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    out: list[tuple[str, str]] = []
    seen_values: set[str] = set()
    for var in _GEMINI_KEY_VARS:
        v = os.environ.get(var)
        if v and v not in seen_values:  # dedupe identical values
            out.append((var, v))
            seen_values.add(v)
    return out


# ---------------------------------------------------------------------------
# OpenAI (GPT-5.x)
# ---------------------------------------------------------------------------

class OpenAIClient:
    """OpenAI Chat Completions client for the GPT-5 family.

    Uses ``/v1/chat/completions`` (the Responses API works too but
    Chat Completions is more familiar and supports the ``response_format``
    JSON-object mode directly).

    Reasoning is controlled via ``reasoning_effort``: ``"minimal"`` for
    no thinking, ``"medium"`` for thinking on. For GPT-5 series the
    output token budget is set with ``max_completion_tokens`` (older
    ``max_tokens`` is not accepted on reasoning models).

    Reasoning trace is *not* exposed in Chat Completions output, so
    ``LLMResponse.thinking`` will be None.
    """

    BASE_URL = "https://api.openai.com/v1"

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        timeout: float = 300.0,
        max_retries: int = 4,
    ) -> None:
        self._model = model
        self._api_key = api_key or _resolve_openai_key()
        self._client = httpx.Client(timeout=timeout)
        self._max_retries = max_retries

    @property
    def model(self) -> str:
        return self._model

    def complete(
        self,
        prompt: str,
        *,
        reasoning: bool = False,
        json_mode: bool = True,
        temperature: float = 1.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        url = f"{self.BASE_URL}/chat/completions"
        body: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            # gpt-5.4 onwards uses 'none', earlier 5-series used 'minimal'
            # for the no-thinking case. We use 'none' since it's the
            # current convention; if running against older 5.0/5.2 series
            # this may need adjustment.
            "reasoning_effort": "medium" if reasoning else "none",
        }
        if max_tokens is not None:
            body["max_completion_tokens"] = max_tokens
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        # Some GPT-5 reasoning models reject explicit temperature; only
        # send it when we actually want non-default behaviour.
        if temperature != 1.0:
            body["temperature"] = temperature

        data = self._post_with_retry(url, body)
        text = ""
        choices = data.get("choices") or []
        if choices:
            msg = choices[0].get("message") or {}
            text = msg.get("content") or ""

        return LLMResponse(
            text=text,
            thinking=None,  # not surfaced in Chat Completions
            model=self._model,
            reasoning_enabled=bool(reasoning),
            raw=data,
        )

    def _post_with_retry(self, url: str, body: dict) -> dict:
        headers = {"Authorization": f"Bearer {self._api_key}"}
        backoff = 2.0
        for attempt in range(self._max_retries + 1):
            r = self._client.post(url, json=body, headers=headers)
            if r.status_code == 200:
                return r.json()
            if (r.status_code in (429, 500, 502, 503, 504)
                    and attempt < self._max_retries):
                time.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
                continue
            r.raise_for_status()
        raise RuntimeError("unreachable")

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "OpenAIClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _resolve_openai_key() -> str:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    for var in ("OPENAI_API_KEY",):
        v = os.environ.get(var)
        if v:
            return v
    raise SystemExit(
        "no OPENAI_API_KEY found in env or .env"
    )


# ---------------------------------------------------------------------------
# DeepSeek (V4 series)
# ---------------------------------------------------------------------------

class DeepSeekClient:
    """DeepSeek V4 Chat Completions client (OpenAI-compatible).

    Hits ``https://api.deepseek.com/v1/chat/completions``. Auth is the
    standard ``Authorization: Bearer <DEEPSEEK_API_KEY>`` header.

    DeepSeek V4 is a *hybrid* thinking model — the same model id serves
    both modes, and reasoning is toggled via the request body rather
    than by swapping model names. We send ``thinking={"type": "enabled"
    | "disabled"}`` per the V4 docs (api-docs.deepseek.com). When
    ``reasoning=True`` we also pass ``reasoning_effort="high"`` to
    encourage non-trivial chains.

    The legacy V3 model names ``deepseek-chat`` and ``deepseek-reasoner``
    are deprecated (sunset 2026-07-24) and currently alias V4-Flash's
    non-thinking / thinking modes respectively. They still work; this
    client passes them through unchanged but does not force the
    ``thinking`` flag for them (the alias already implies the mode).
    Prefer ``deepseek-v4-flash`` / ``deepseek-v4-pro`` directly.

    JSON output: V4 supports ``response_format={"type": "json_object"}``.
    The DeepSeek docs caution that the prompt must literally contain the
    word "json" or the API may return empty/whitespace content; the
    cloze prompt template already mentions JSON so this is fine in
    practice, but callers using novel prompts should keep that in mind.

    Token budget: uses ``max_tokens`` (not ``max_completion_tokens``).

    Thinking trace: when reasoning is enabled, the assistant message
    object carries a ``reasoning_content`` field alongside ``content``;
    we surface it as ``LLMResponse.thinking``.
    """

    BASE_URL = "https://api.deepseek.com/v1"

    # Legacy V3 aliases. These already encode the mode in the model name,
    # so we don't try to override with a `thinking` flag.
    _LEGACY_MODELS = frozenset({"deepseek-chat", "deepseek-reasoner"})

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        timeout: float = 300.0,
        max_retries: int = 4,
    ) -> None:
        self._model = model
        self._api_key = api_key or _resolve_deepseek_key()
        self._client = httpx.Client(timeout=timeout)
        self._max_retries = max_retries

    @property
    def model(self) -> str:
        return self._model

    def complete(
        self,
        prompt: str,
        *,
        reasoning: bool = False,
        json_mode: bool = True,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        url = f"{self.BASE_URL}/chat/completions"
        body: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        # V4 hybrid models: toggle via `thinking` parameter. Legacy
        # aliases already encode the mode in the model name.
        if self._model not in self._LEGACY_MODELS:
            body["thinking"] = {
                "type": "enabled" if reasoning else "disabled",
            }
            if reasoning:
                body["reasoning_effort"] = "high"
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        data = self._post_with_retry(url, body)

        text = ""
        thinking: str | None = None
        choices = data.get("choices") or []
        if choices:
            msg = choices[0].get("message") or {}
            text = msg.get("content") or ""
            rc = msg.get("reasoning_content")
            if rc:
                thinking = rc

        return LLMResponse(
            text=text,
            thinking=thinking,
            model=self._model,
            reasoning_enabled=bool(reasoning),
            raw=data,
        )

    def _post_with_retry(self, url: str, body: dict) -> dict:
        headers = {"Authorization": f"Bearer {self._api_key}"}
        backoff = 2.0
        for attempt in range(self._max_retries + 1):
            r = self._client.post(url, json=body, headers=headers)
            if r.status_code == 200:
                return r.json()
            if (r.status_code in (429, 500, 502, 503, 504)
                    and attempt < self._max_retries):
                time.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
                continue
            r.raise_for_status()
        raise RuntimeError("unreachable")

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "DeepSeekClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _resolve_deepseek_key() -> str:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    v = os.environ.get("DEEPSEEK_API_KEY")
    if v:
        return v
    raise SystemExit(
        "no DEEPSEEK_API_KEY found in env or .env"
    )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_client(model: str, *, timeout: float = 300.0) -> LLMClient:
    """Return the appropriate client for the given model name.

    - ``gemini-*``         → Gemini (Google AI Studio)
    - ``deepseek-*``       → DeepSeek (api.deepseek.com)
    - ``gpt-*`` / ``o*``   → OpenAI (Chat Completions)
    - everything else      → Ollama (qwen, llama, gpt-oss, etc.)

    Note: ``gpt-oss-*`` and any name containing ``/`` are treated as
    Ollama (open-weight / local). DeepSeek dispatch happens *after* the
    Ollama-prefix check so that ``user/deepseek-…`` style local names
    still route to Ollama.
    """
    if model.startswith("gemini"):
        return GeminiClient(model, timeout=timeout)
    if model.startswith("gpt-oss") or "/" in model:
        return OllamaClient(model, timeout=timeout)
    if model.startswith("deepseek"):
        return DeepSeekClient(model, timeout=timeout)
    if model.startswith("gpt-") or model.startswith("o1") or model.startswith("o3"):
        return OpenAIClient(model, timeout=timeout)
    return OllamaClient(model, timeout=timeout)


# ---------------------------------------------------------------------------
# Response parsing helpers
# ---------------------------------------------------------------------------

_CODE_FENCE_RE = re.compile(
    r"^```(?:json|JSON)?\s*\n?|\n?```\s*$",
    re.MULTILINE,
)


def _strip_fences(text: str) -> str:
    return _CODE_FENCE_RE.sub("", text).strip()


def _extract_first_object(text: str) -> str | None:
    """Find the first balanced top-level JSON object in `text`."""
    depth = 0
    start = -1
    in_str = False
    escape = False
    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_str:
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                return text[start:i + 1]
    return None


def parse_answer(response: LLMResponse) -> dict[str, Any] | None:
    """Pull the ``answer`` object out of the model's JSON response.

    Tolerates a few common deviations from the requested schema:
    - Markdown `````json`` fences around the JSON.
    - Leading or trailing prose around the JSON object.

    Returns the ``answer`` dict when one can be parsed out, and
    ``None`` otherwise---specifically: no parseable JSON object, no
    ``answer`` key, or an ``answer`` whose value is not a dict (e.g.\
    the model returned ``{"answer": "Some title"}``; that single
    string cannot be mapped onto the requested field-to-value schema,
    so it is treated as a parse failure rather than wrapped). Downstream
    scorers score ``None`` as "all wrong" for every queried field.
    """
    text = (response.text or "").strip()
    if not text:
        return None
    text = _strip_fences(text)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        snippet = _extract_first_object(text)
        if not snippet:
            return None
        try:
            obj = json.loads(snippet)
        except json.JSONDecodeError:
            return None
    if not isinstance(obj, dict):
        return None
    answer = obj.get("answer")
    return answer if isinstance(answer, dict) else None
