## Calm AI

Minimal FastAPI + Streamlit project for:

- generating synthetic daily wellness logs
- training a simple risk classifier
- predicting low/medium/high risk
- producing supportive, non-clinical recommendations

### Prerequisites

- **Python 3.10+** installed and available as `python` or `python3`
- A terminal: **Terminal** or **iTerm** on macOS; **PowerShell** or **Command Prompt** on Windows

Open a terminal, `cd` into this project folder, then follow **Setup** for your OS.

### Setup

#### macOS (and Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

#### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If activation fails with an execution-policy error, run once for your user account, then activate again:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

On Windows **Command Prompt** (cmd), use `.\.venv\Scripts\activate.bat` instead of `Activate.ps1`.

After setup, keep the virtual environment **activated** for the commands below. On macOS/Linux use `python3` where this doc shows `python` on Windows, or use `python` if that is what your venv provides.

### Generate synthetic data

**macOS / Linux**

```bash
python3 scripts/generate_synthetic_data.py --users 500 --days 60 --seed 42
```

**Windows**

```powershell
python scripts\generate_synthetic_data.py --users 500 --days 60 --seed 42
```

### Train the model

**macOS / Linux**

```bash
python3 scripts/train_model.py
```

**Windows**

```powershell
python scripts\train_model.py
```

This writes `app/ml/artifacts/risk_model.joblib`. The API needs this file before `POST /ml/predict` can succeed.

### Run the API

**macOS / Linux**

```bash
uvicorn app.main:app --reload
```

**Windows**

```powershell
uvicorn app.main:app --reload
```

The API listens at **http://127.0.0.1:8000** by default.

Useful endpoints:

- `GET /health`
- `POST /ml/predict`
- `POST /recommendations`

Example predict request (**macOS / Linux** — bash):

```bash
curl -X POST "http://127.0.0.1:8000/ml/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "mood": "okay",
    "stress": 5.2,
    "craving": 4.1,
    "sleep_hours": 7.0,
    "exercise_minutes": 20,
    "social_interaction": 30,
    "trigger_boredom": 0,
    "trigger_loneliness": 0,
    "trigger_conflict": 0,
    "days_since_last_relapse": 7
  }'
```

**Windows (PowerShell):** use `curl.exe` so you invoke the real curl client, not the PowerShell alias:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/ml/predict" -H "Content-Type: application/json" -d '{"mood":"okay","stress":5.2,"craving":4.1,"sleep_hours":7.0,"exercise_minutes":20,"social_interaction":30,"trigger_boredom":0,"trigger_loneliness":0,"trigger_conflict":0,"days_since_last_relapse":7}'
```

### Run the Streamlit UI

Use a **second** terminal (venv activated, project folder as the current directory). The UI calls the API at `http://127.0.0.1:8000` by default (configurable in the sidebar).

**macOS / Linux**

```bash
streamlit run frontend/streamlit_app.py
```

**Windows**

```powershell
streamlit run frontend\streamlit_app.py
```

### Typical workflow

1. Activate the venv and install dependencies (see **Setup**).
2. Generate data and train the model (**Generate synthetic data**, **Train the model**).
3. Start the API in one terminal (`uvicorn app.main:app --reload`).
4. Start Streamlit in another terminal (**Run the Streamlit UI**).
