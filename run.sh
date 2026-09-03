#!/usr/bin/env bash
# ==============================================================================
# SARAI: Safety & Alignment Red-teaming AI Evaluation Studio
# Startup & Dependency Installation Script
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "🛡️  SARAI Evaluation Studio: Setup & Launch"
echo "============================================================"

# ── 1. Python & Virtual Environment Setup ─────────────────────────────────────
PYTHON_BIN=""
for py_cmd in python3 python python3.13 python3.12 python3.11; do
    if command -v "$py_cmd" >/dev/null 2>&1; then
        PYTHON_BIN="$py_cmd"
        break
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "❌ Error: Python 3 was not found. Please install Python 3.10+ and re-run."
    exit 1
fi

echo "[1/4] Using Python: $($PYTHON_BIN --version)"

VENV_DIR="$SCRIPT_DIR/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "[2/4] Creating virtual environment at ./venv..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
else
    echo "[2/4] Virtual environment found at ./venv."
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# ── 2. Dependency Installation ────────────────────────────────────────────────
echo "[3/4] Installing / verifying dependencies from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

# ── 3. Gemini API Key Configuration (Secure - No Logging) ────────────────────
echo "[4/4] Configuring Gemini API Key..."

# Disable streamlit telemetry by default
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

CURRENT_KEY="${GEMINI_API_KEY:-$GOOGLE_API_KEY}"

if [ -n "$CURRENT_KEY" ]; then
    echo "  ↳ Existing API key detected in environment."
    read -r -p "    Do you want to use the existing API key? [Y/n]: " use_existing_choice
    case "$use_existing_choice" in
        [nN][oO]|[nN])
            CURRENT_KEY=""
            ;;
        *)
            echo "    [✓] Using existing API key."
            ;;
    esac
fi

if [ -z "$CURRENT_KEY" ]; then
    while [ -z "$CURRENT_KEY" ]; do
        # -s flag prevents typing from being echoed to the terminal or captured in logs
        read -s -r -p "    Enter your Gemini API key (hidden): " entered_key
        echo ""
        entered_key="$(echo "$entered_key" | tr -d '[:space:]')"
        if [ -n "$entered_key" ]; then
            CURRENT_KEY="$entered_key"
            echo "    [✓] API key securely set in session."
        else
            echo "    ⚠️ API key cannot be empty. Please enter a valid key."
        fi
    done
fi

export GEMINI_API_KEY="$CURRENT_KEY"
export GOOGLE_API_KEY="$CURRENT_KEY"
unset CURRENT_KEY
unset entered_key

# ── 4. Launch Streamlit Application ───────────────────────────────────────────
APP_PORT="${PORT:-8501}"

echo ""
echo "============================================================"
echo "🚀 SARAI Evaluation Studio is ready!"
echo "📍 Access Local Testing URL: http://localhost:${APP_PORT}"
echo "   (Press Ctrl+C to stop the server)"
echo "============================================================"
echo ""

exec streamlit run streamlit_app.py \
    --server.port "$APP_PORT" \
    --server.headless true \
    --browser.gatherUsageStats false
