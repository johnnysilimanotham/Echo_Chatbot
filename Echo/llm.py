"""LLM integration helpers for Echo.

This module is intentionally dependency-light: it uses `requests` and an
OpenAI-compatible Chat Completions API so Echo can work with local LLMs
(e.g., Ollama) or hosted providers without changing code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class LLMConfig:
	base_url: str
	model: str
	api_key: str | None = None
	temperature: float = 0.7
	max_tokens: int = 220
	timeout_s: float = 30.0


def _truthy(value: str | None) -> bool:
	if value is None:
		return False
	return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def load_llm_config_from_env() -> LLMConfig | None:
	"""Load LLM settings from env vars.

	Env vars:
	- ECHO_USE_LLM: if truthy, enables LLM calls
	- ECHO_LLM_BASE_URL: OpenAI-compatible base URL (default: Ollama's /v1)
	- ECHO_LLM_MODEL: required model name
	- ECHO_LLM_API_KEY: optional bearer token
	- ECHO_LLM_TEMPERATURE: optional float
	- ECHO_LLM_MAX_TOKENS: optional int
	- ECHO_LLM_TIMEOUT_S: optional float
	"""

	if not _truthy(os.getenv("ECHO_USE_LLM")):
		return None

	base_url = (os.getenv("ECHO_LLM_BASE_URL") or "http://localhost:11434/v1").strip()
	model = (os.getenv("ECHO_LLM_MODEL") or "").strip()
	if not model:
		return None

	api_key = (os.getenv("ECHO_LLM_API_KEY") or "").strip() or None

	temperature_raw = (os.getenv("ECHO_LLM_TEMPERATURE") or "").strip()
	max_tokens_raw = (os.getenv("ECHO_LLM_MAX_TOKENS") or "").strip()
	timeout_raw = (os.getenv("ECHO_LLM_TIMEOUT_S") or "").strip()

	temperature = 0.7
	if temperature_raw:
		try:
			temperature = float(temperature_raw)
		except ValueError:
			pass

	max_tokens = 220
	if max_tokens_raw:
		try:
			max_tokens = int(max_tokens_raw)
		except ValueError:
			pass

	timeout_s = 30.0
	if timeout_raw:
		try:
			timeout_s = float(timeout_raw)
		except ValueError:
			pass

	return LLMConfig(
		base_url=base_url,
		model=model,
		api_key=api_key,
		temperature=temperature,
		max_tokens=max_tokens,
		timeout_s=timeout_s,
	)


def chat_completion(config: LLMConfig, messages: list[dict[str, str]]) -> str:
	"""Call an OpenAI-compatible Chat Completions endpoint and return content."""

	url = config.base_url.rstrip("/") + "/chat/completions"
	headers: dict[str, str] = {"Content-Type": "application/json"}
	if config.api_key:
		headers["Authorization"] = f"Bearer {config.api_key}"

	payload: dict[str, Any] = {
		"model": config.model,
		"messages": messages,
		"temperature": config.temperature,
		"max_tokens": config.max_tokens,
	}

	response = requests.post(url, headers=headers, json=payload, timeout=config.timeout_s)
	response.raise_for_status()
	data = response.json()

	choices = data.get("choices") or []
	if not choices:
		raise ValueError("LLM response contained no choices.")

	message = (choices[0] or {}).get("message") or {}
	content = (message.get("content") or "").strip()
	if not content:
		raise ValueError("LLM response contained empty content.")

	return content

