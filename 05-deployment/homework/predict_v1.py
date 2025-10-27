import pickle

model_file = 'pipeline_v1.bin'
with open(model_file, 'rb') as f_in:
    pipeline = pickle.load(f_in)


data = {
    "lead_source": "organic_search",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0
}

pred = pipeline.predict_proba(data)[0, 1]
print(f"V1 model prediction: {pred:.3f}")