import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_auc_score

import pickle

# TRAINING PARAMS
n_splits = 5
C = 1.0

# DATA PREPROCESSING
url = "https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(url)

df.columns = df.columns.str.lower().str.replace(' ', '_')

categorical_columns = df.select_dtypes(exclude='number').columns

for c in categorical_columns:
	df[c] = df[c].str.lower().str.replace(' ', '_')

df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)

df.churn = (df.churn == 'yes').astype(int)

df.drop('customerid', axis=1, inplace=True)

# MODEL TRAINING
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

def one_hot_encoding(df, enc=None):
	df.reset_index(drop=True, inplace=True)

	df_categorical = df.select_dtypes(exclude='number')
	df_numerical = df.select_dtypes('number')

	if not enc:
		enc = OneHotEncoder(sparse_output=False, dtype=np.int32)
		enc.fit(df_categorical)

	df_encoded = pd.DataFrame(
		data=enc.transform(df_categorical),
		columns=enc.get_feature_names_out())

	X = pd.concat([df_encoded, df_numerical], ignore_index=True, axis=1)

	return X.values, enc

def train(df, y, C=1.0):
	# encode the data
	X, enc = one_hot_encoding(df)

	# train a model
	model = LogisticRegression(C=C, max_iter=10000)
	model.fit(X, y)

	# return encoder and model
	return enc, model


def predict(df, enc, model):
	# encode the data
	X, _ = one_hot_encoding(df, enc)

	# generate prediction probabilities
	pred = model.predict_proba(X)[:, 1]

	return pred


kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

scores = []

print("Model training...")

for train_idx, val_idx in kfold.split(df_full_train):
	# randomly split training and validation sets
	df_train = df_full_train.iloc[train_idx]
	df_val = df_full_train.iloc[val_idx]

	y_train = df_train.churn.values
	y_val = df_val.churn.values

	df_train = df_train.drop('churn', axis=1)
	df_val = df_val.drop('churn', axis=1)

	# train and validate the model
	enc, model = train(df_train, y_train, C=C)
	y_pred = predict(df_val, enc, model)

	auc = roc_auc_score(y_val, y_pred)
	scores.append(auc)

print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))

# MODEL VALIDATION
y_full_train = df_full_train.churn.values
y_test = df_test.churn.values

df_full_train = df_full_train.drop('churn', axis=1)
df_test = df_test.drop('churn', axis=1)

enc, model = train(df_full_train, y_full_train, C=1.0)

y_pred = predict(df_test, enc, model)
auc = roc_auc_score(y_test, y_pred)
print(f"Validation AUC: {auc:.3f}")

# MODEL SAVING
output_file = f'model_C={C}.bin'

with open(output_file, 'wb') as f_out:
	pickle.dump((enc, model), f_out)

print(f"Model saved to '{output_file}'")