## Calm AI

Minimal FastAPI + Streamlit project for:
- generating synthetic daily wellness logs
- training a simple risk classifier
- predicting low/medium/high risk
- producing supportive, non-clinical recommendations

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

### Generate synthetic data

```bash
python3 scripts/generate_synthetic_data.py --users 500 --days 60 --seed 42
```

### Train the model

```bash
python3 scripts/train_model.py
```

This saves the artifact to `app/ml/artifacts/risk_model.joblib`.

### Run the API

```bash
uvicorn app.main:app --reload
```

Useful endpoints:
- `GET /health`
- `POST /ml/predict`
- `POST /recommendations`

Example predict request:

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

### Run the Streamlit UI

```bash
streamlit run frontend/streamlit_app.py
```

