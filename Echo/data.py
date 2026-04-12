"""Static data used by the Echo mental health chatbot."""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd


SYSTEM_MESSAGE = (
	"Echo is a supportive mental health companion, not a licensed clinician. "
	"Echo provides emotional support, grounding ideas, and self-care prompts."
)


CRISIS_KEYWORDS = {
	"suicide",
	"kill myself",
	"end my life",
	"self harm",
	"self-harm",
	"hurt myself",
	"overdose",
	"can't go on",
	"want to die",
}


EMOTION_LIBRARY = {
	"anxiety": [
		"Thanks for sharing that. Anxiety can feel overwhelming. Would a 60-second breathing reset help right now?",
		"It sounds like your nervous system is on high alert. Let's slow things down together with one small grounding step.",
	],
	"stress": [
		"That sounds like a lot to carry. What's one pressure point we can make 10% lighter today?",
		"You're holding a lot right now. If we break this into one tiny next step, what feels manageable?",
	],
	"sadness": [
		"I'm really glad you reached out. Sadness can make everything feel heavier than usual.",
		"That sounds painful. You do not have to carry this by yourself in this moment.",
	],
	"lonely": [
		"Feeling alone can be deeply hard. You matter, and I'm here with you right now.",
		"Loneliness can make the world feel quiet and distant. Want to find one gentle way to reconnect today?",
	],
	"burnout": [
		"Burnout is real, and it can numb motivation. Rest is not weakness, it is repair.",
		"Your energy sounds depleted. Let's choose one task to pause and one need to prioritize.",
	],
}


GENERIC_SUPPORT = [
	"Thank you for opening up. I'm here to listen and support you.",
	"You're not alone in this. We can take it one step at a time.",
	"I hear you. Let's focus on what feels most important right now.",
]


COPING_STRATEGIES = [
	"Try the 5-4-3-2-1 grounding exercise: name 5 things you see, 4 you feel, 3 you hear, 2 you smell, 1 you taste.",
	"Breathe in for 4 seconds, hold for 4, out for 6. Repeat for one minute.",
	"Write one kind sentence to yourself that you would say to a friend in your situation.",
	"Take a 2-minute reset: unclench your jaw, drop your shoulders, and drink some water.",
]


FOLLOW_UP_QUESTIONS = [
	"Would you like a grounding exercise, journaling prompt, or a small action plan?",
	"Do you want to talk more about what triggered this feeling today?",
	"What would feeling 5% better in the next hour look like for you?",
]


def random_pick(options: list[str]) -> str:
	"""Return one random option from a non-empty list."""
	return random.choice(options)


DATA_PATH_CANDIDATES = [
	Path(__file__).resolve().parent.parent / "mental_health_dataset.csv",
	Path(__file__).resolve().parent.parent / "Data.csv",
]

CATEGORICAL_COLUMNS = [
	"gender",
	"region",
	"income_level",
	"education_level",
	"daily_role",
	"device_type",
]

NUMERIC_COLUMNS = [
	"id",
	"age",
	"device_hours_per_day",
	"phone_unlocks",
	"notifications_per_day",
	"social_media_mins",
	"study_mins",
	"physical_activity_days",
	"sleep_hours",
	"sleep_quality",
	"anxiety_score",
	"depression_score",
	"stress_level",
	"happiness_score",
	"focus_score",
	"high_risk_flag",
	"productivity_score",
	"digital_dependence_score",
]


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
	"""Apply practical cleaning steps to the Echo CSV dataset."""
	if df.empty:
		return df

	clean_df = df.copy()
	clean_df.columns = clean_df.columns.str.strip()
	clean_df = clean_df.drop_duplicates(keep="first")

	# Generic cleanup for any background dataset schema.
	for column in clean_df.columns:
		if pd.api.types.is_object_dtype(clean_df[column]) or pd.api.types.is_string_dtype(clean_df[column]):
			clean_df[column] = clean_df[column].astype("string").str.strip()

	for column in clean_df.columns:
		if column in CATEGORICAL_COLUMNS:
			continue
		if pd.api.types.is_string_dtype(clean_df[column]):
			converted = pd.to_numeric(clean_df[column], errors="coerce")
			non_null_original = clean_df[column].notna().sum()
			non_null_converted = converted.notna().sum()
			# Convert string columns when most values are numeric-like.
			if non_null_original > 0 and (non_null_converted / non_null_original) >= 0.8:
				clean_df[column] = converted

	for column in CATEGORICAL_COLUMNS:
		if column in clean_df.columns:
			clean_df[column] = clean_df[column].astype("string").str.strip()

	for column in NUMERIC_COLUMNS:
		if column in clean_df.columns:
			clean_df[column] = pd.to_numeric(clean_df[column], errors="coerce")

	if "id" in clean_df.columns:
		clean_df = clean_df.drop_duplicates(subset=["id"], keep="first")

	numeric_subset = [column for column in NUMERIC_COLUMNS if column in clean_df.columns]
	generic_numeric_columns = clean_df.select_dtypes(include=["number"]).columns.tolist()
	numeric_subset = sorted(set(numeric_subset + generic_numeric_columns))
	if numeric_subset:
		clean_df[numeric_subset] = clean_df[numeric_subset].fillna(clean_df[numeric_subset].median())

	for column in CATEGORICAL_COLUMNS:
		if column in clean_df.columns:
			mode = clean_df[column].mode(dropna=True)
			if not mode.empty:
				clean_df[column] = clean_df[column].fillna(mode.iloc[0])

	if "age" in clean_df.columns:
		clean_df["age"] = clean_df["age"].clip(lower=10, upper=100)
	if "physical_activity_days" in clean_df.columns:
		clean_df["physical_activity_days"] = clean_df["physical_activity_days"].clip(lower=0, upper=7)
	if "sleep_hours" in clean_df.columns:
		clean_df["sleep_hours"] = clean_df["sleep_hours"].clip(lower=0, upper=24)
	if "high_risk_flag" in clean_df.columns:
		clean_df["high_risk_flag"] = (clean_df["high_risk_flag"] > 0).astype(int)

	if "id" in clean_df.columns:
		clean_df["id"] = clean_df["id"].round().astype(int)
		clean_df = clean_df.sort_values("id").reset_index(drop=True)

	return clean_df


def load_dataset(csv_path: Path | None = None) -> pd.DataFrame:
	"""Read and preprocess the Echo dataset CSV."""
	if csv_path is not None:
		if not csv_path.exists():
			return pd.DataFrame()
		raw_df = pd.read_csv(csv_path)
		return preprocess_dataset(raw_df)

	for candidate in DATA_PATH_CANDIDATES:
		if candidate.exists():
			raw_df = pd.read_csv(candidate)
			return preprocess_dataset(raw_df)

	return pd.DataFrame()


DATASET = load_dataset()
