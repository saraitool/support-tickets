# 🛡️ SARAI: Safety & Alignment Red-teaming AI Evaluation Studio

An agentic end-to-end framework for AI safety benchmarking, taxonomy generation, red-teaming evaluation, automated LLM-as-a-judge autorating, and interactive error analysis.

---

### Quickstart

To install dependencies, securely configure your API key, and launch the application:

```bash
./run.sh
```

The script will:
1. Detect or create a virtual environment (`./venv`).
2. Install all required dependencies (`requirements.txt`).
3. Securely prompt for your Gemini API key (using masked input with zero terminal logging).
4. Launch Streamlit with telemetry disabled and display your local testing URL: `http://localhost:8501`.

---

### Manual Setup (Alternative)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY="your-gemini-api-key"
streamlit run streamlit_app.py --browser.gatherUsageStats false
```


