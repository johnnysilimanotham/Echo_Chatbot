"""Core response logic for the Echo mental health chatbot."""

from __future__ import annotations

from typing import Any

from data import (
	COPING_STRATEGIES,
	CRISIS_KEYWORDS,
	EMOTION_LIBRARY,
	FOLLOW_UP_QUESTIONS,
	GENERIC_SUPPORT,
	SYSTEM_MESSAGE,
	random_pick,
)

from llm import chat_completion, load_llm_config_from_env


def _normalize(text: str) -> str:
	return text.strip().lower()


def _contains_crisis_language(message: str) -> bool:
	normalized = _normalize(message)
	return any(keyword in normalized for keyword in CRISIS_KEYWORDS)


def _detect_emotion_bucket(message: str) -> str | None:
	normalized = _normalize(message)
	if any(word in normalized for word in ["anxious", "anxiety", "panic", "worried", "overthinking"]):
		return "anxiety"
	if any(word in normalized for word in ["stressed", "pressure", "overwhelmed", "deadline"]):
		return "stress"
	if any(word in normalized for word in ["sad", "down", "depressed", "hopeless", "empty"]):
		return "sadness"
	if any(word in normalized for word in ["alone", "lonely", "isolated", "disconnected"]):
		return "lonely"
	if any(word in normalized for word in ["burned out", "burnout", "exhausted", "drained"]):
		return "burnout"
	return None


def crisis_response() -> str:
	return (
		"I'm really glad you said this out loud. Your safety matters most right now. "
		"Please contact local emergency services immediately if you may act on these thoughts. "
		"If you're in the U.S. or Canada, call or text 988 for the Suicide & Crisis Lifeline. "
		"If you're elsewhere, I can help you find the right local crisis line."
	)


def _coerce_chat_history(conversation: Any) -> list[dict[str, str]]:
	if not isinstance(conversation, list):
		return []
	coerced: list[dict[str, str]] = []
	for item in conversation:
		if not isinstance(item, dict):
			continue
		role = item.get("role")
		content = item.get("content")
		if role in {"system", "user", "assistant"} and isinstance(content, str) and content.strip():
			coerced.append({"role": role, "content": content.strip()})
	return coerced


def _build_llm_messages(history: list[dict[str, str]], user_message: str) -> list[dict[str, str]]:
	system_prompt = (
		f"{SYSTEM_MESSAGE}\n\n"
		"Guidelines:\n"
		"- Be warm, supportive, and practical.\n"
		"- Keep it concise (roughly 3-8 sentences).\n"
		"- Offer 1 grounding/coping suggestion.\n"
		"- Ask 1 gentle follow-up question.\n"
		"- Do not claim to be a therapist or to diagnose.\n"
	)

	trimmed_history = history[-12:] if history else []

	# Ensure the latest user message is present (streamlit already appends it).
	if not trimmed_history or trimmed_history[-1].get("role") != "user" or trimmed_history[-1].get("content") != user_message:
		trimmed_history = [*trimmed_history, {"role": "user", "content": user_message}]

	return [{"role": "system", "content": system_prompt}, *trimmed_history]


def _try_llm_response(user_message: str, conversation: Any = None) -> str | None:
	config = load_llm_config_from_env()
	if config is None:
		return None

	history = _coerce_chat_history(conversation)
	messages = _build_llm_messages(history, user_message)
	try:
		return chat_completion(config, messages)
	except Exception:
		return None


def generate_echo_response(user_message: str, conversation: Any = None) -> str:
	"""Generate a supportive response for a user message.

	If env var `ECHO_USE_LLM` is enabled and LLM settings are present, Echo will
	try to use an OpenAI-compatible LLM for responses; otherwise it falls back to
	the built-in template responses.
	"""
	if not user_message.strip():
		return "I'm here with you. Share whatever is on your mind, even if it's hard to put into words."

	if _contains_crisis_language(user_message):
		return crisis_response()

	llm_reply = _try_llm_response(user_message, conversation=conversation)
	if llm_reply:
		return llm_reply

	emotion_key = _detect_emotion_bucket(user_message)
	if emotion_key:
		support = random_pick(EMOTION_LIBRARY[emotion_key])
	else:
		support = random_pick(GENERIC_SUPPORT)

	coping = random_pick(COPING_STRATEGIES)
	follow_up = random_pick(FOLLOW_UP_QUESTIONS)

	return f"{support}\n\n{coping}\n\n{follow_up}"
