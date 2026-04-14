"""Streamlit interface for the Echo mental health chatbot."""

from __future__ import annotations

import json
import os
from datetime import date, datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd
import streamlit as st

from data import SYSTEM_MESSAGE
from functions import generate_echo_response

CHECKIN_FILE = Path(__file__).resolve().parent / "checkins.json"
PARTS_OF_DAY = ["Morning", "Afternoon", "Evening", "Night"]
MOOD_OPTIONS = ["Very low", "Low", "Neutral", "Good", "Great"]
MOOD_TO_SCORE = {
	"Very low": 1,
	"Low": 2,
	"Neutral": 3,
	"Good": 4,
	"Great": 5,
}
MOOD_TO_COLOR = {
	"Very low": "#d62828",
	"Low": "#f77f00",
	"Neutral": "#8d99ae",
	"Good": "#2a9d8f",
	"Great": "#2b9348",
}


def _reset_chat() -> None:
	st.session_state.messages = [
		{
			"role": "assistant",
			"content": "Hi, I'm Echo. I'm here to support you. How are you feeling today?",
		}
	]


def _chat_transcript(messages: list[dict[str, str]]) -> str:
	lines: list[str] = []
	for msg in messages:
		role = (msg.get("role") or "").strip().lower()
		content = (msg.get("content") or "").strip()
		if not content:
			continue
		prefix = "You" if role == "user" else "Echo"
		lines.append(f"{prefix}: {content}")
	return ("\n\n".join(lines).strip() + "\n") if lines else ""


def _load_checkins() -> list[dict[str, str]]:
	if not CHECKIN_FILE.exists():
		return []
	try:
		return json.loads(CHECKIN_FILE.read_text(encoding="utf-8"))
	except (json.JSONDecodeError, OSError):
		return []


def _save_checkins(entries: list[dict[str, str]]) -> None:
	CHECKIN_FILE.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def _sort_checkins(entries: list[dict[str, str]]) -> list[dict[str, str]]:
	return sorted(entries, key=lambda x: (x.get("date", ""), x.get("time", "")), reverse=True)


def _ensure_entry_ids(entries: list[dict[str, str]]) -> bool:
	changed = False
	for entry in entries:
		if not entry.get("id"):
			entry["id"] = uuid4().hex
			changed = True
	return changed


def _streak_days(entries: list[dict[str, str]]) -> int:
	if not entries:
		return 0

	entry_dates = set()
	for entry in entries:
		entry_date = entry.get("date")
		if not entry_date:
			continue
		try:
			entry_dates.add(date.fromisoformat(entry_date))
		except ValueError:
			continue

	if not entry_dates:
		return 0

	streak = 0
	day_cursor = date.today()
	while day_cursor in entry_dates:
		streak += 1
		day_cursor = day_cursor.fromordinal(day_cursor.toordinal() - 1)

	return streak


def _missing_parts_today(entries: list[dict[str, str]]) -> list[str]:
	today_key = date.today().isoformat()
	today_parts = {
		entry.get("part_of_day")
		for entry in entries
		if entry.get("date") == today_key and entry.get("part_of_day")
	}
	return [part for part in PARTS_OF_DAY if part not in today_parts]


def _weekly_mood_trend(entries: list[dict[str, str]]) -> pd.DataFrame:
	rows = []
	for entry in entries:
		entry_date = entry.get("date")
		mood = entry.get("mood")
		if not entry_date or mood not in MOOD_TO_SCORE:
			continue
		try:
			parsed_date = date.fromisoformat(entry_date)
		except ValueError:
			continue
		week_start = parsed_date.fromordinal(parsed_date.toordinal() - parsed_date.weekday())
		rows.append({"week_start": week_start, "mood_score": MOOD_TO_SCORE[mood]})

	if not rows:
		return pd.DataFrame(columns=["week_start", "avg_mood"])

	trend = pd.DataFrame(rows).groupby("week_start", as_index=False).mean(numeric_only=True)
	trend = trend.rename(columns={"mood_score": "avg_mood"}).sort_values("week_start")
	return trend


def _mood_chip_html(mood: str) -> str:
	color = MOOD_TO_COLOR.get(mood, "#6c757d")
	return (
		"<span style='display:inline-block;"
		"padding:0.15rem 0.55rem;"
		"border-radius:999px;"
		"font-size:0.82rem;"
		"font-weight:700;"
		"color:#ffffff;"
		f"background:{color};'>{mood}</span>"
	)


st.set_page_config(page_title="Echo | Mental Health Chatbot", page_icon="💬", layout="centered")

st.markdown(
	"""
	<style>
		.main {
			background: linear-gradient(180deg, #f8fbff 0%, #eef5ef 100%);
		}
		.echo-title {
			font-size: 2.2rem;
			font-weight: 700;
			margin-bottom: 0.25rem;
			color: #1f3b2c;
		}
		.echo-subtitle {
			color: #405a48;
			margin-top: 0;
			margin-bottom: 1rem;
		}
		.echo-notice {
			border: 1px solid #bfd5c3;
			background: #f4faf5;
			border-radius: 12px;
			padding: 0.75rem 0.9rem;
			color: #23422f;
			margin-bottom: 1rem;
		}
	</style>
	""",
	unsafe_allow_html=True,
)

st.markdown('<p class="echo-title">Echo</p>', unsafe_allow_html=True)
st.markdown(
	'<p class="echo-subtitle">A gentle space to reflect, regulate, and reset.</p>',
	unsafe_allow_html=True,
)

st.markdown(
	f'<div class="echo-notice"><strong>Important:</strong> {SYSTEM_MESSAGE} '
	"If you are in immediate danger, call your local emergency number now.</div>",
	unsafe_allow_html=True,
)

if "checkins" not in st.session_state:
	st.session_state.checkins = _load_checkins()
	if _ensure_entry_ids(st.session_state.checkins):
		st.session_state.checkins = _sort_checkins(st.session_state.checkins)
		_save_checkins(st.session_state.checkins)

if "edit_checkin_id" not in st.session_state:
	st.session_state.edit_checkin_id = None

if "messages" not in st.session_state:
	st.session_state.messages = [
		{
			"role": "assistant",
			"content": (
				"Hi, I'm Echo. I'm here to support you. "
				"How are you feeling today?"
			),
		}
	]

chat_tab, journal_tab = st.tabs(["Chat", "Daily Check-ins"])

with st.sidebar:
	st.subheader("Chat")
	if st.button("New conversation"):
		_reset_chat()
		st.rerun()

	transcript = _chat_transcript(st.session_state.messages)
	st.download_button(
		"Download chat",
		data=transcript,
		file_name="echo_chat.txt",
		mime="text/plain",
		disabled=not transcript,
	)

	st.divider()
	st.subheader("LLM (optional)")
	env_model = (os.getenv("ECHO_LLM_MODEL") or "").strip()
	env_enabled = (os.getenv("ECHO_USE_LLM") or "").strip().lower() in {"1", "true", "yes", "y", "on"}
	use_llm = st.checkbox("Use LLM responses", value=bool(env_enabled and env_model))
	llm_model = st.text_input("Model", value=env_model, placeholder="e.g. llama3.1")
	llm_base_url = st.text_input(
		"Base URL",
		value=(os.getenv("ECHO_LLM_BASE_URL") or "http://localhost:11434/v1").strip(),
		help="Must expose an OpenAI-compatible POST /chat/completions endpoint.",
	)
	llm_api_key = st.text_input(
		"API key",
		value=(os.getenv("ECHO_LLM_API_KEY") or "").strip(),
		type="password",
		help="Optional for local LLMs; required by some hosted providers.",
	)

	if use_llm and not llm_model.strip():
		st.warning("Enter a model name to enable LLM responses.")

	# Apply settings for this Streamlit session (Echo reads from env).
	os.environ["ECHO_USE_LLM"] = "1" if (use_llm and llm_model.strip()) else "0"
	os.environ["ECHO_LLM_MODEL"] = llm_model.strip()
	os.environ["ECHO_LLM_BASE_URL"] = llm_base_url.strip()
	if llm_api_key.strip():
		os.environ["ECHO_LLM_API_KEY"] = llm_api_key.strip()
	else:
		os.environ.pop("ECHO_LLM_API_KEY", None)

with chat_tab:
	for message in st.session_state.messages:
		with st.chat_message(message["role"]):
			st.markdown(message["content"])

	user_input = st.chat_input("Share what is on your mind...")

	if user_input:
		st.session_state.messages.append({"role": "user", "content": user_input})
		with st.chat_message("user"):
			st.markdown(user_input)

		reply = generate_echo_response(user_input, conversation=st.session_state.messages)
		st.session_state.messages.append({"role": "assistant", "content": reply})

		with st.chat_message("assistant"):
			st.markdown(reply)

with journal_tab:
	streak = _streak_days(st.session_state.checkins)
	st.metric("Current check-in streak", f"{streak} day(s)")

	today_key = date.today().isoformat()
	today_count = sum(1 for entry in st.session_state.checkins if entry.get("date") == today_key)
	missing_parts = _missing_parts_today(st.session_state.checkins)
	if today_count == 0:
		st.warning("Reminder: you have not logged a check-in today yet.")
	elif missing_parts:
		st.info(f"You still have open check-in windows today: {', '.join(missing_parts)}.")
	else:
		st.success("Great consistency. You completed all day-part check-ins today.")

	with st.form("checkin_form", clear_on_submit=True):
		col1, col2 = st.columns(2)
		with col1:
			entry_date = st.date_input("Date", value=date.today())
		with col2:
			entry_time = st.time_input("Time", value=datetime.now().time().replace(second=0, microsecond=0))

		part_of_day = st.selectbox(
			"Part of day",
			PARTS_OF_DAY,
		)
		mood = st.selectbox(
			"Mood",
			MOOD_OPTIONS,
		)
		note = st.text_area("Short journal note", placeholder="What happened, and how did it feel?")

		submitted = st.form_submit_button("Save check-in")

	if submitted:
		entry = {
			"id": uuid4().hex,
			"date": entry_date.isoformat(),
			"time": entry_time.strftime("%H:%M"),
			"part_of_day": part_of_day,
			"mood": mood,
			"note": note.strip(),
		}
		st.session_state.checkins.append(entry)
		st.session_state.checkins = _sort_checkins(st.session_state.checkins)
		_save_checkins(st.session_state.checkins)
		st.success("Check-in saved.")

	if st.session_state.checkins:
		st.subheader("Weekly mood trend")
		trend_df = _weekly_mood_trend(st.session_state.checkins)
		if not trend_df.empty:
			chart_df = trend_df.set_index("week_start")
			st.line_chart(chart_df["avg_mood"], y_label="Average mood (1-5)", x_label="Week")
		else:
			st.caption("Not enough data yet for a weekly mood trend chart.")

		# Export check-ins in one click.
		export_df = pd.DataFrame(st.session_state.checkins)
		export_cols = ["id", "date", "time", "part_of_day", "mood", "note"]
		export_df = export_df[[col for col in export_cols if col in export_df.columns]]
		st.download_button(
			"Export check-ins (CSV)",
			data=export_df.to_csv(index=False),
			file_name="echo_checkins.csv",
			mime="text/csv",
		)

		st.subheader("Calendar-style timeline")
		for day in sorted({entry["date"] for entry in st.session_state.checkins}, reverse=True):
			with st.expander(day):
				for entry in [item for item in st.session_state.checkins if item["date"] == day]:
					entry_id = entry["id"]
					controls_col, edit_col, delete_col = st.columns([7, 1, 1])
					with controls_col:
						st.markdown(f"**{entry['time']}** | {entry['part_of_day']}")
						st.markdown(_mood_chip_html(entry["mood"]), unsafe_allow_html=True)
						if entry.get("note"):
							st.caption(entry["note"])
					with edit_col:
						if st.button("Edit", key=f"edit_btn_{entry_id}"):
							st.session_state.edit_checkin_id = entry_id
					with delete_col:
						if st.button("Del", key=f"delete_btn_{entry_id}"):
							st.session_state.checkins = [
								item for item in st.session_state.checkins if item.get("id") != entry_id
							]
							_save_checkins(st.session_state.checkins)
							if st.session_state.edit_checkin_id == entry_id:
								st.session_state.edit_checkin_id = None
							st.rerun()

					if st.session_state.edit_checkin_id == entry_id:
						try:
							default_date = date.fromisoformat(entry["date"])
						except ValueError:
							default_date = date.today()
						try:
							default_time = datetime.strptime(entry["time"], "%H:%M").time()
						except ValueError:
							default_time = datetime.now().time().replace(second=0, microsecond=0)

						with st.form(f"edit_form_{entry_id}"):
							edit_col1, edit_col2 = st.columns(2)
							with edit_col1:
								edit_date = st.date_input("Edit date", value=default_date, key=f"edit_date_{entry_id}")
							with edit_col2:
								edit_time = st.time_input("Edit time", value=default_time, key=f"edit_time_{entry_id}")

							edit_part = st.selectbox(
								"Edit part of day",
								PARTS_OF_DAY,
								index=PARTS_OF_DAY.index(entry["part_of_day"]) if entry["part_of_day"] in PARTS_OF_DAY else 0,
								key=f"edit_part_{entry_id}",
							)
							edit_mood = st.selectbox(
								"Edit mood",
								MOOD_OPTIONS,
								index=MOOD_OPTIONS.index(entry["mood"]) if entry["mood"] in MOOD_OPTIONS else 2,
								key=f"edit_mood_{entry_id}",
							)
							edit_note = st.text_area(
								"Edit note",
								value=entry.get("note", ""),
								key=f"edit_note_{entry_id}",
							)

							save_edit = st.form_submit_button("Update entry")
							cancel_edit = st.form_submit_button("Cancel")

						if save_edit:
							entry["date"] = edit_date.isoformat()
							entry["time"] = edit_time.strftime("%H:%M")
							entry["part_of_day"] = edit_part
							entry["mood"] = edit_mood
							entry["note"] = edit_note.strip()
							st.session_state.checkins = _sort_checkins(st.session_state.checkins)
							_save_checkins(st.session_state.checkins)
							st.session_state.edit_checkin_id = None
							st.rerun()

						if cancel_edit:
							st.session_state.edit_checkin_id = None
							st.rerun()
	else:
		st.info("No check-ins yet. Add your first mood journal entry above.")
