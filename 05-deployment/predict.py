import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import pickle

C = 1.0

output_file = f"model_C={C}.bin"

with open(output_file, 'rb') as f_in:
	dv, model = pickle.load(f_in)


customer = {
	'customerid': '6946-lmsqs',
	'gender': 'male',
	'seniorcitizen': 1,
	'partner': 'yes',
	'dependents': 'no',
	'tenure': 25,
	'phoneservice': 'yes',
	'multiplelines': 'yes',
	'internetservice': 'fiber_optic',
	'onlinesecurity': 'yes',
	'onlinebackup': 'no',
	'deviceprotection': 'no',
	'techsupport': 'no',
	'streamingtv': 'no',
	'streamingmovies': 'yes',
	'contract': 'one_year',
	'paperlessbilling': 'yes',
	'paymentmethod': 'electronic_check',
	'monthlycharges': 89.05,
	'totalcharges': 2177.45
}

X = dv.transform([customer])
prediction_prob = model.predict_proba(X)[0, 1]

print(f'Churn probability: {prediction_prob:.3f}')