"""Core response logic for the Echo mental health chatbot."""

from __future__ import annotations

from data import (
	COPING_STRATEGIES,
	CRISIS_KEYWORDS,
	EMOTION_LIBRARY,
	FOLLOW_UP_QUESTIONS,
	GENERIC_SUPPORT,
	random_pick,
)


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


def generate_echo_response(user_message: str) -> str:
	"""Generate a supportive response for a user message."""
	if not user_message.strip():
		return "I'm here with you. Share whatever is on your mind, even if it's hard to put into words."

	if _contains_crisis_language(user_message):
		return crisis_response()

	emotion_key = _detect_emotion_bucket(user_message)
	if emotion_key:
		support = random_pick(EMOTION_LIBRARY[emotion_key])
	else:
		support = random_pick(GENERIC_SUPPORT)

	coping = random_pick(COPING_STRATEGIES)
	follow_up = random_pick(FOLLOW_UP_QUESTIONS)

	return f"{support}\n\n{coping}\n\n{follow_up}"
