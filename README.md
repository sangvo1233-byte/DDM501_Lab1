# Lab 1 — Credit Default Risk Scoring API

DDM501 — AI in DevOps, DataOps, MLOps · Lab 1

The service lets the loan origination system call the model that estimates the probability that a
credit card customer misses their next payment. It turns that probability into an underwriting
decision.

```
probability < 0.30          ->  LOW     ->  APPROVE
0.30 <= probability < 0.60  ->  MEDIUM  ->  REVIEW   (a human underwriter looks)
probability >= 0.60         ->  HIGH    ->  DECLINE
```

Both thresholds are business decisions. They live in `app/config.py` and can be overridden with the
`REVIEW_THRESHOLD` and `DECLINE_THRESHOLD` environment variables.

## Project layout

| Path | What it is |
|---|---|
| `app/schemas.py` | The API contract: field bounds, closed value sets, two custom validators |
| `app/model.py` | Loads the artifact, flattens the payload into the 23 training columns, applies the decision rule |
| `app/main.py` | FastAPI app: `/health`, `/predict`, `/predict/batch`, `/model/info` |
| `scripts/train_model.py` | Trains the HistGradientBoosting pipeline and writes `models/credit_model.joblib` |
| `tests/test_api.py` | 28 tests: happy path, model behaviour, validation, batching |
| `Dockerfile`, `docker-compose.yml` | Container that runs anywhere, with the model mounted read-only |

## Run it

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/train_model.py        # ROC AUC ~0.748, writes models/credit_model.joblib
pytest tests/ -v --cov=app --cov-report=term-missing
uvicorn app.main:app --reload        # http://localhost:8000/docs
```

With Docker:

```bash
python scripts/train_model.py        # the model is mounted, not baked into the image
docker compose up --build -d
docker inspect --format='{{.State.Health.Status}}' credit-risk-api   # -> healthy
```

## Example

```bash
curl -X POST http://localhost:8000/predict -H 'Content-Type: application/json' -d '{
  "limit_bal": 300000, "sex": 2, "education": 1, "marriage": 2, "age": 38,
  "pay_status": [-1,-1,-1,-1,-1,-1],
  "bill_amt":  [12000,11500,11000,10500,10000,9500],
  "pay_amt":   [12000,11500,11000,10500,10000,9500]}'
```

```json
{"default_probability": 0.0637, "risk_band": "LOW", "decision": "APPROVE",
 "review_threshold": 0.3, "decline_threshold": 0.6, "model_version": "1.0.0"}
```

## Implementation notes

- **Contract at the edge.** `sex`, `education` and `marriage` are `Literal` types. `age` must be 18–100. The three
  monthly lists must have exactly 6 entries each. Two `field_validator`s cover what `Field` cannot express:
  every `pay_status` value must be in −2..8, and no `pay_amt` value may be negative. A malformed request gets a
  422 and never reaches the model.
- **No training–serving skew.** `to_frame` rebuilds the exact 23 columns in `FEATURE_COLUMNS` order, including
  the dataset's missing `PAY_1` (`pay_status[0..5]` maps to `PAY_0, PAY_2..PAY_6`).
- **Decision order.** `decide` checks the DECLINE threshold first. Checking REVIEW first would send every
  high-risk applicant to manual review instead of declining it.
- **503 vs 500.** If the model is not loaded, `/predict` returns 503 because the request was valid and the
  service is not ready. Unexpected scoring failures are logged and return 500.
- **Batch is vectorised.** `/predict/batch` calls `score_batch` once for up to 500 applications, and
  `test_matches_single_prediction` proves the batch path gives exactly what `/predict` gives.
- **Container.** Dependencies are installed before the source is copied, for layer caching. The container runs
  as non-root `appuser` (uid 1000), and uvicorn is bound to `0.0.0.0`. The health check calls `/health` and fails
  unless `model_loaded` is `true`, so a live process with no model is reported unhealthy.

## Results

| Check | Result |
|---|---|
| Model (HistGradientBoosting, held-out 20%) | ROC AUC 0.748, PR AUC 0.546 (baseline 0.233) |
| Tests | 28 passed, coverage 90% |
| Docker | image builds and the container reports `healthy` (see `docs/evidence/`) |
