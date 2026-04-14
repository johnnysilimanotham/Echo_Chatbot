# Echo Chatbot

Echo is a Streamlit-based mental health support chatbot with optional LLM-backed responses.

## Run locally

```powershell
pip install -r requirements.txt
streamlit run Echo/streamlit_app.py
```

## Optional: enable an LLM

By default, Echo uses the built-in (non-LLM) response templates in `Echo/functions.py`.

To enable LLM responses, set these environment variables:

```powershell
$env:ECHO_USE_LLM="1"
$env:ECHO_LLM_MODEL="your-model-name"
# Optional (defaults to Ollama's OpenAI-compatible API):
$env:ECHO_LLM_BASE_URL="http://localhost:11434/v1"
# Optional (only if your provider requires it):
$env:ECHO_LLM_API_KEY="..."
```

Echo expects an **OpenAI-compatible** `POST /chat/completions` endpoint at `ECHO_LLM_BASE_URL`.

