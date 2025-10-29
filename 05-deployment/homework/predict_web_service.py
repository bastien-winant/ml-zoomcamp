from fastapi import FastAPI
import pickle
from pydantic import BaseModel
from flask import jsonify

class Campaign(BaseModel):
	lead_source: str
	number_of_courses_viewed: int
	annual_income: float

model_file = 'pipeline_v1.bin'

with open(model_file, 'rb') as f_in:
	model = pickle.load(f_in)

app = FastAPI()

@app.post('/predict')
def predict(campaign: Campaign):
	pred = model.predict_proba(campaign.model_dump())[0, 1]
	conversion = pred >= 0.5

	result = {
		'conversion_probability': float(pred),
		'conversion': bool(conversion)
	}

	return result