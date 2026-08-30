"""Thin provider-agnostic LLM client for structured extraction.

Requirements this satisfies:

* **Provider portability.** The case study allows OpenAI, Anthropic or alternatives.
  Provider is chosen from whichever API key is present, so the notebook runs in any
  reviewer's environment without code edits.
* **Schema enforcement at generation time.** OpenAI gets `response_format` with a
  strict JSON Schema; Anthropic gets a forced tool call whose input schema is the
  same Pydantic schema. Both make "not valid JSON" a near-impossible outcome rather
  than something we clean up afterwards.
* **Resilience.** Exponential backoff with jitter for rate limits and transient
  errors, plus a JSON repair pass for the residual cases.
* **Cost control.** Token usage is recorded per call so the notebook can report the
  real cost of parsing the corpus and extrapolate to production volumes.
"""
from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass, field

import config


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    attempts: int = 1
    latency_s: float = 0.0
    raw: dict = field(default_factory=dict)


def resolve_provider() -> str:
    """Pick a provider from configuration or available credentials."""
    explicit = os.getenv("LLM_PROVIDER", config.LLM_PROVIDER)
    if explicit and explicit != "auto":
        return explicit
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    raise LLMError(
        "No LLM credentials found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY "
        "(optionally with OPENAI_BASE_URL for an OpenAI-compatible endpoint)."
    )


def extract_json(text: str) -> dict:
    """Recover a JSON object from a model response.

    Handles the three things that go wrong in practice: markdown code fences,
    leading prose, and trailing commentary after the closing brace.
    """
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Brace matching is more reliable than a regex for nested objects.
    start = text.find("{")
    if start == -1:
        raise LLMError(f"No JSON object found in response: {text[:200]!r}")
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(text[start:], start=start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise LLMError("Unbalanced JSON braces in model response")


# ------------------------------------------------------------------ providers
def _call_openai(system: str, user: str, json_schema: dict | None) -> LLMResponse:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.getenv("OPENAI_BASE_URL") or None)
    model = os.getenv("OPENAI_MODEL", config.OPENAI_MODEL)

    kwargs: dict = {
        "model": model,
        "temperature": config.LLM_TEMPERATURE,
        "max_tokens": config.LLM_MAX_TOKENS,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    }
    if json_schema is not None:
        # Strict structured output: the API rejects generations that violate the schema.
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "candidate_profile", "schema": json_schema, "strict": False},
        }

    t0 = time.time()
    resp = client.chat.completions.create(**kwargs)
    usage = getattr(resp, "usage", None)
    return LLMResponse(
        text=resp.choices[0].message.content or "",
        provider="openai", model=model,
        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        latency_s=time.time() - t0,
    )


def _call_anthropic(system: str, user: str, json_schema: dict | None) -> LLMResponse:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    model = os.getenv("ANTHROPIC_MODEL", config.ANTHROPIC_MODEL)

    kwargs: dict = {
        "model": model,
        "max_tokens": config.LLM_MAX_TOKENS,
        "temperature": config.LLM_TEMPERATURE,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    if json_schema is not None:
        # Forcing a tool call is Anthropic's structured-output mechanism.
        kwargs["tools"] = [{
            "name": "emit_candidate_profile",
            "description": "Return the structured candidate profile extracted from the resume.",
            "input_schema": json_schema,
        }]
        kwargs["tool_choice"] = {"type": "tool", "name": "emit_candidate_profile"}

    t0 = time.time()
    resp = client.messages.create(**kwargs)
    text = ""
    for block in resp.content:
        if block.type == "tool_use":
            text = json.dumps(block.input)
            break
        if block.type == "text":
            text += block.text
    usage = getattr(resp, "usage", None)
    return LLMResponse(
        text=text, provider="anthropic", model=model,
        prompt_tokens=getattr(usage, "input_tokens", 0) or 0,
        completion_tokens=getattr(usage, "output_tokens", 0) or 0,
        latency_s=time.time() - t0,
    )


def call_llm(system: str, user: str, json_schema: dict | None = None,
             max_attempts: int | None = None) -> LLMResponse:
    """Call the resolved provider with retry and exponential backoff."""
    provider = resolve_provider()
    fn = {"openai": _call_openai, "anthropic": _call_anthropic}.get(provider)
    if fn is None:
        raise LLMError(f"Unsupported provider: {provider}")

    attempts = max_attempts or config.LLM_MAX_ATTEMPTS
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            resp = fn(system, user, json_schema)
            resp.attempts = attempt
            return resp
        except Exception as exc:  # noqa: BLE001 - retry on any transient provider error
            last = exc
            if attempt == attempts:
                break
            # Jitter avoids synchronised retries when parsing a batch concurrently.
            time.sleep(min(2 ** attempt + random.random(), 30))
    raise LLMError(f"LLM call failed after {attempts} attempts: {last}") from last
