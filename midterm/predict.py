import pickle

expedition = {
	'year': 2006,
	'season': 'Autumn',
	'host': 'China',
	'nation': 'Italy',
	'camps': 2,
	'totmembers': 1,
	'tothired': 0,
	'o2climb': False,
	'o2descent': False,
	'o2sleep': False,
	'comrte': True,
	'stdrte': True,
	'primrte': False,
	'sponsored': True,
	'ascent_route': 'CHOY-NW_side',
	'total_experience': 15,
	'total_leadership_experience': 8,
	'total_successes': 10,
	'avg_climber_age': 57.0,
	'leaders': 1,
	'support': 0,
	'disabled': 0,
	'hired': 0,
	'sherpas': 0
}

file_name = "criterion_gini__max_depth_20__min_samples_leaf_15.bin"
with open(file_name, 'rb') as f_in:
	dv, model = pickle.load(f_in)
	X = dv.transform([expedition])
	pred = model.predict_proba(X)[0, 1]

	print(f"Expedition failure probability: {pred:.3f}")