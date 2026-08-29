import io
import os
import zipfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pytorch_tabular import TabularModel

app = FastAPI(title="Cardiovascular Risk Prediction API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_ZIP = BASE_DIR / "ft_transformer_model.zip"
MODEL_DIR = BASE_DIR / "model_artifacts"
PIPELINE_PATH = BASE_DIR / "preprocessing_pipeline.pkl"
EXPECTED_FEATURES = ["male", "age", "education", "currentSmoker", "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes", "BMI"]
MODEL_FEATURES = ["male", "age", "education", "currentSmoker", "BPMeds", "prevalentStroke", "prevalentHyp", "BMI"]
DISCLAIMER = "Research prototype for cardiovascular risk screening. This result is not a medical diagnosis and should not replace professional medical advice."

model: TabularModel | None = None
pipeline: dict[str, Any] | None = None
artifact_error: str | None = None


def load_artifacts() -> None:
    global model, pipeline, artifact_error
    try:
        if not PIPELINE_PATH.exists() or not MODEL_ZIP.exists():
            raise RuntimeError("Saved model artifacts are missing from the application.")
        pipeline = joblib.load(PIPELINE_PATH)
        if not isinstance(pipeline, dict) or not all(k in pipeline for k in ("imputer", "power_transformer", "scaler", "feature_order", "selected_features")):
            raise RuntimeError("The saved preprocessing pipeline has an unsupported structure.")
        MODEL_DIR.mkdir(exist_ok=True)
        if not (MODEL_DIR / "model.ckpt").exists():
            with zipfile.ZipFile(MODEL_ZIP) as archive:
                archive.extractall(MODEL_DIR)
        model = TabularModel.load_model(str(MODEL_DIR), map_location="cpu")
        artifact_error = None
    except Exception as exc:
        model = None
        pipeline = None
        artifact_error = str(exc)
        print(f"Artifact loading failed: {exc}")


load_artifacts()


class PatientInput(BaseModel):
    male: int = Field(..., ge=0, le=1)
    age: float = Field(..., ge=18, le=120)
    education: float = Field(1, ge=1, le=4)
    currentSmoker: int = Field(..., ge=0, le=1)
    BPMeds: int = Field(..., ge=0, le=1)
    prevalentStroke: int = Field(..., ge=0, le=1)
    prevalentHyp: int = Field(..., ge=0, le=1)
    diabetes: int = Field(..., ge=0, le=1)
    BMI: float = Field(..., ge=10, le=70)


def ensure_ready() -> None:
    if model is None or pipeline is None:
        raise HTTPException(status_code=503, detail="The saved prediction model is not ready. Please try again shortly.")


def prepare_features(source: pd.DataFrame) -> pd.DataFrame:
    ensure_ready()
    missing = [name for name in EXPECTED_FEATURES if name not in source.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required columns: {', '.join(missing)}")
    frame = source[EXPECTED_FEATURES].copy()
    for name in EXPECTED_FEATURES:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    if frame.isna().all(axis=0).any():
        bad = [name for name in EXPECTED_FEATURES if frame[name].isna().all()]
        raise HTTPException(status_code=400, detail=f"These columns contain no valid numeric values: {', '.join(bad)}")
    try:
        order = pipeline["feature_order"]
        transformed = pipeline["imputer"].transform(frame[order])
        transformed = pipeline["power_transformer"].transform(transformed)
        transformed = pipeline["scaler"].transform(transformed)
        prepared = pd.DataFrame(transformed, columns=order, index=frame.index)[pipeline["selected_features"]]
        if not np.isfinite(prepared.to_numpy()).all():
            raise ValueError("Preprocessing produced a non-finite value")
        return prepared[MODEL_FEATURES]
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"The submitted values could not be processed: {exc}") from exc


def predict_frame(source: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    prepared = prepare_features(source)
    try:
        output = model.predict(prepared, include_input_features=False, progress_bar=None)
        probability_column = "target_1_probability"
        if probability_column not in output.columns:
            candidates = [c for c in output.columns if c.endswith("_probability")]
            probability_column = candidates[-1] if candidates else ""
        if not probability_column:
            raise ValueError("The loaded checkpoint did not return class probabilities.")
        probabilities = np.clip(output[probability_column].to_numpy(dtype=float), 0, 1)
        result = source.copy()
        result["predicted_probability"] = np.round(probabilities, 6)
        result["risk_category"] = np.where(probabilities >= 0.5, "Higher Risk", "Lower Risk")
        return result, probabilities
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction could not be completed: {exc}") from exc


@app.get("/api/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None, "pipeline_loaded": pipeline is not None, "expected_features": EXPECTED_FEATURES, "artifact_error": artifact_error}


@app.post("/api/predict/manual")
def predict_manual(data: PatientInput):
    source = pd.DataFrame([data.model_dump()])
    result, probabilities = predict_frame(source)
    probability = float(probabilities[0])
    return {"predicted_probability": probability, "risk_percentage": round(probability * 100, 2), "risk_category": str(result.iloc[0]["risk_category"]), "disclaimer": DISCLAIMER}


@app.post("/api/predict/dataset")
async def predict_dataset(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")
    try:
        source = pd.read_csv(io.BytesIO(await file.read()))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"The CSV could not be read: {exc}") from exc
    if source.empty:
        raise HTTPException(status_code=400, detail="The uploaded CSV is empty.")
    result, probabilities = predict_frame(source)
    records = []
    for number, (_, row) in enumerate(result.head(100).iterrows(), 1):
        records.append({"record": number, "predicted_probability": float(row["predicted_probability"]), "risk_category": str(row["risk_category"])})
    stream = io.StringIO()
    result.to_csv(stream, index=False)
    return {"total_records": len(result), "higher_risk_count": int((result["risk_category"] == "Higher Risk").sum()), "lower_risk_count": int((result["risk_category"] == "Lower Risk").sum()), "average_predicted_risk": float(np.mean(probabilities)), "records": records, "csv_data": stream.getvalue(), "filename": f"predicted_{file.filename}"}