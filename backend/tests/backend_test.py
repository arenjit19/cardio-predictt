"""Backend regression tests for Cardiovascular Risk Prediction API.

Tests cover:
- /api/health artifact loading state
- /api/predict/manual valid + invalid inputs
- /api/predict/dataset valid multi-row CSV, missing columns, invalid file
- Response schema and data assertions (probabilities, categories, disclaimer)
- Downloaded CSV contains predictions
"""
import io
import os
import csv
import pytest
import requests

BASE_URL = os.environ["REACT_APP_BACKEND_URL"].rstrip("/")


@pytest.fixture(scope="module")
def api():
    session = requests.Session()
    return session


# ---------- /api/health ----------
class TestHealth:
    def test_health_ok(self, api):
        r = api.get(f"{BASE_URL}/api/health", timeout=30)
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["pipeline_loaded"] is True
        assert data["artifact_error"] is None
        assert data["expected_features"] == [
            "male", "age", "education", "currentSmoker", "BPMeds",
            "prevalentStroke", "prevalentHyp", "diabetes", "BMI",
        ]


# ---------- /api/predict/manual ----------
VALID_MANUAL = {
    "male": 1, "age": 55.0, "education": 2.0, "currentSmoker": 1,
    "BPMeds": 0, "prevalentStroke": 0, "prevalentHyp": 1,
    "diabetes": 0, "BMI": 28.5,
}


class TestManualPrediction:
    def test_valid_manual_returns_real_probability(self, api):
        r = api.post(f"{BASE_URL}/api/predict/manual", json=VALID_MANUAL, timeout=60)
        assert r.status_code == 200, r.text
        data = r.json()
        assert set(data.keys()) >= {"predicted_probability", "risk_percentage", "risk_category", "disclaimer"}
        p = data["predicted_probability"]
        assert isinstance(p, float)
        assert 0.0 <= p <= 1.0
        # Not a placeholder-flat number
        assert p not in (0.0, 1.0), f"suspect placeholder probability={p}"
        assert data["risk_percentage"] == round(p * 100, 2)
        assert data["risk_category"] in ("Higher Risk", "Lower Risk")
        assert (data["risk_category"] == "Higher Risk") == (p >= 0.5)
        assert "not a medical diagnosis" in data["disclaimer"].lower()

    def test_manual_low_risk_profile(self, api):
        low = {"male": 0, "age": 25, "education": 4, "currentSmoker": 0,
               "BPMeds": 0, "prevalentStroke": 0, "prevalentHyp": 0,
               "diabetes": 0, "BMI": 22.0}
        r = api.post(f"{BASE_URL}/api/predict/manual", json=low, timeout=60)
        assert r.status_code == 200
        assert 0.0 <= r.json()["predicted_probability"] <= 1.0

    def test_manual_high_risk_profile(self, api):
        high = {"male": 1, "age": 78, "education": 1, "currentSmoker": 1,
                "BPMeds": 1, "prevalentStroke": 1, "prevalentHyp": 1,
                "diabetes": 1, "BMI": 34.0}
        r = api.post(f"{BASE_URL}/api/predict/manual", json=high, timeout=60)
        assert r.status_code == 200
        assert 0.0 <= r.json()["predicted_probability"] <= 1.0

    def test_manual_probability_differs_across_profiles(self, api):
        low = {"male": 0, "age": 25, "education": 4, "currentSmoker": 0,
               "BPMeds": 0, "prevalentStroke": 0, "prevalentHyp": 0,
               "diabetes": 0, "BMI": 22.0}
        high = {"male": 1, "age": 78, "education": 1, "currentSmoker": 1,
                "BPMeds": 1, "prevalentStroke": 1, "prevalentHyp": 1,
                "diabetes": 1, "BMI": 34.0}
        r1 = api.post(f"{BASE_URL}/api/predict/manual", json=low, timeout=60).json()
        r2 = api.post(f"{BASE_URL}/api/predict/manual", json=high, timeout=60).json()
        assert r1["predicted_probability"] != r2["predicted_probability"], "Model produced identical probabilities for very different profiles"

    def test_manual_invalid_age_low(self, api):
        payload = dict(VALID_MANUAL, age=5)
        r = api.post(f"{BASE_URL}/api/predict/manual", json=payload, timeout=30)
        assert r.status_code == 422

    def test_manual_invalid_bmi_high(self, api):
        payload = dict(VALID_MANUAL, BMI=500)
        r = api.post(f"{BASE_URL}/api/predict/manual", json=payload, timeout=30)
        assert r.status_code == 422

    def test_manual_missing_field(self, api):
        payload = dict(VALID_MANUAL)
        payload.pop("BMI")
        r = api.post(f"{BASE_URL}/api/predict/manual", json=payload, timeout=30)
        assert r.status_code == 422


# ---------- /api/predict/dataset ----------
COLUMNS = ["male", "age", "education", "currentSmoker", "BPMeds",
           "prevalentStroke", "prevalentHyp", "diabetes", "BMI"]

ROWS = [
    [1, 55, 2, 1, 0, 0, 1, 0, 28.5],
    [0, 40, 3, 0, 0, 0, 0, 0, 24.1],
    [1, 68, 1, 1, 1, 0, 1, 1, 31.2],
    [0, 30, 4, 0, 0, 0, 0, 0, 21.5],
    [1, 72, 1, 1, 1, 1, 1, 1, 33.4],
]


def build_csv(cols, rows):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(cols)
    for r in rows:
        w.writerow(r)
    return buf.getvalue().encode("utf-8")


class TestDatasetPrediction:
    def test_valid_multi_row_csv(self, api):
        csv_bytes = build_csv(COLUMNS, ROWS)
        r = api.post(
            f"{BASE_URL}/api/predict/dataset",
            files={"file": ("test.csv", csv_bytes, "text/csv")},
            timeout=60,
        )
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["total_records"] == len(ROWS)
        assert data["higher_risk_count"] + data["lower_risk_count"] == len(ROWS)
        assert 0.0 <= data["average_predicted_risk"] <= 1.0
        assert len(data["records"]) == len(ROWS)
        for rec in data["records"]:
            assert 0.0 <= rec["predicted_probability"] <= 1.0
            assert rec["risk_category"] in ("Higher Risk", "Lower Risk")
        # csv_data contains original columns + predicted_probability + risk_category
        assert data["csv_data"].startswith(",".join(COLUMNS)) or all(
            c in data["csv_data"].splitlines()[0] for c in COLUMNS
        )
        header = data["csv_data"].splitlines()[0]
        for c in COLUMNS + ["predicted_probability", "risk_category"]:
            assert c in header, f"missing column {c} in downloaded csv"
        assert data["filename"].endswith(".csv")

    def test_missing_columns(self, api):
        cols = ["male", "age", "BMI"]
        rows = [[1, 55, 28.5]]
        csv_bytes = build_csv(cols, rows)
        r = api.post(
            f"{BASE_URL}/api/predict/dataset",
            files={"file": ("bad.csv", csv_bytes, "text/csv")},
            timeout=30,
        )
        assert r.status_code == 400
        detail = r.json().get("detail", "")
        assert "Missing required columns" in detail

    def test_empty_csv(self, api):
        r = api.post(
            f"{BASE_URL}/api/predict/dataset",
            files={"file": ("empty.csv", b"", "text/csv")},
            timeout=30,
        )
        assert r.status_code == 400

    def test_non_csv_extension(self, api):
        r = api.post(
            f"{BASE_URL}/api/predict/dataset",
            files={"file": ("bad.txt", b"whatever", "text/plain")},
            timeout=30,
        )
        assert r.status_code == 400
        assert "CSV" in r.json()["detail"]

    def test_non_numeric_column(self, api):
        # BMI column filled with non-numeric only -> should error
        cols = COLUMNS
        rows = [[1, 55, 2, 1, 0, 0, 1, 0, "abc"]]
        csv_bytes = build_csv(cols, rows)
        r = api.post(
            f"{BASE_URL}/api/predict/dataset",
            files={"file": ("bad2.csv", csv_bytes, "text/csv")},
            timeout=30,
        )
        # Backend detects columns with no valid numeric values
        assert r.status_code == 400
