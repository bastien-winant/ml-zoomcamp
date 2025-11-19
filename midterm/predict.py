import pickle
from flask import Flask, request, jsonify

model_file = "criterion_gini__max_depth_20__min_samples_leaf_15.bin"

with open(model_file, 'rb') as f_in:
	dv, model = pickle.load(f_in)

app = Flask('expedition_pred')

@app.route('/predict', methods=['POST'])
def predict():
	expedition = request.get_json()
	X = dv.transform([expedition])
	pred = model.predict_proba(X)[0, 1]
	failure_pred = pred >= .5

	result = {
		'failure_probability': float(pred),
		'failure': bool(failure_pred)
	}

	return jsonify(result)


if __name__ == "__main__":
	app.run(debug=True, host='0.0.0.0', port=9696)