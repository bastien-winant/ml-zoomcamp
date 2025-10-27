from fastapi import FastAPI, Body
import pickle
import pandas as pd

app = FastAPI()

# Load model
model_file = 'pipeline_v1.bin'
with open(model_file, 'rb') as f_in:
    pipeline = pickle.load(f_in)

@app.post("/predict")
def predict(data: dict = Body(...)):
    X = pd.DataFrame([data])
    pred_proba = pipeline.predict_proba(X)[0, 1]
    churn = pred_proba >= 0.5
    return {
        "churn_probability": float(pred_proba),
        "churn": bool(churn)
    }
