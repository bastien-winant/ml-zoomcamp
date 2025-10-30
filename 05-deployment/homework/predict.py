import pickle

model_file = 'pipeline_v1.bin'

with open(model_file, 'rb') as f_in:
	pipeline = pickle.load(f_in)


X = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}

pred = pipeline.predict_proba(X)[0, 1]
print(f"Model conversion prediction: {pred:.3f}")