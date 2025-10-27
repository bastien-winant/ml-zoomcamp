import pickle
from flask import Flask, request, jsonify

C = 1.0

output_file = f"model_C={C}.bin"
with open(output_file, 'rb') as f_in:
	dv, model = pickle.load(f_in)


app = Flask('churn')

@app.route('/predict', methods=['POST'])
def predict():
	customer = request.get_json()

	X = dv.transform([customer])
	pred = model.predict_proba(X)[0, 1]
	churn = pred >= .5

	result = {
		'churn_probability': float(pred),
		'churn': bool(churn)
	}
	return jsonify(result)


if __name__ == "__main__":
	app.run(debug=True, host='0.0.0.0', port=9696)