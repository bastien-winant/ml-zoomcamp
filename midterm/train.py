from dbfread import DBF
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings('ignore')

def read_dbf(file_path):
	# read the raw data and store in a dataframe
	dbf = DBF(file_path)
	df = pd.DataFrame(iter(dbf))

	# identify empty strings as missing values
	df.replace("", np.nan, inplace=True)

	# ensure there are now empty rows or columns
	df.dropna(how='all', axis=0, inplace=True)
	df.dropna(how='all', axis=1, inplace=True)

	# remove any duplicates
	df.drop_duplicates(inplace=True)

	# standardize column names
	df.columns = map(lambda x: x.lower(), df.columns)

	return df

exped_df = read_dbf('data/raw/exped.DBF')

# verify that there is still exactly one row per expedition
exped_df.expid = exped_df.expid.str.cat(exped_df.year)
assert exped_df.expid.nunique() == exped_df.shape[0]

exped_df.year = exped_df.year.astype(int)

host_map = {
	0: 'Unknown',
	1: 'Nepal',
	2: 'China',
	3: 'India'
}
exped_df.host = exped_df.host.map(host_map)

season_map = {
	0: 'Unknown',
	1: 'Spring',
	2: 'Summer',
	3: 'Autumn',
	4: 'Winter'
}
exped_df.season = exped_df.season.map(season_map)

# remove expedition with undefined main route
exped_df = exped_df.loc[exped_df.route1.notna()]

# remove expeditions that include non-climbing activities
exped_df = exped_df.loc[~exped_df.traverse & ~exped_df.ski & ~exped_df.parapente]

exped_df.drop(['traverse', 'ski', 'parapente'], axis=1, inplace=True)

# filter based on expedition termination reason:
# 12 - Did not attempt climb
# 13 - Attempt rumored
exped_df = exped_df.loc[~exped_df.termreason.isin([12, 13])]

exped_df.drop('termreason', axis=1, inplace=True)

# remove unused columns
exped_df.drop([
	'route2', 'route3', 'route4', 'success2', 'success3', 'success4', 'ascent2', 'ascent3', 'ascent4', 'claimed',
	'disputed', 'approach', 'smtdate', 'smttime', 'smtdays', 'totdays', 'termdate', 'termnote', 'highpoint',
	'smtmembers', 'mdeaths', 'smthired', 'hdeaths', 'othersmts', 'campsites', 'routememo', 'accidents', 'achievment',
	'primmem', 'primref', 'primid', 'chksum', 'leaders', 'countries', 'ascent1', 'bcdate', 'o2used', 'o2none',
	'o2medical', 'o2unkwn', 'agency', 'o2taken', 'nohired', 'rope'], axis=1, inplace=True)

# create flag variables to indicate whether the expedition has a sponsor
exped_df['sponsored'] = exped_df.sponsor.notna()
exped_df.drop('sponsor', axis=1, inplace=True)

# concatenate peak and route
exped_df['ascent_route'] = exped_df.peakid.str.cat(exped_df.route1, sep='-').str.replace(" ", "_")
exped_df.drop(['peakid', 'route1'], axis=1, inplace=True)

# find most commonly attempted ascents
route_counts = pd.DataFrame(exped_df.ascent_route.value_counts()).reset_index()
common_routes = route_counts.loc[route_counts['count'] >= 10, 'ascent_route']

# keep only expeditions on common routes
exped_df = exped_df.loc[exped_df.ascent_route.isin(common_routes)]

# keep only recent expeditions
exped_df = exped_df.loc[exped_df.year >= 1980]

exped_df.comrte = exped_df.comrte.fillna(False)
exped_df.stdrte = exped_df.stdrte.fillna(False)

exped_df.rename({'success1': 'success'}, axis=1, inplace=True)

### Climber data
climber_df = read_dbf('data/raw/members.DBF')

climber_df = climber_df[[
	'expid', 'myear', 'mseason', 'fname', 'lname', 'yob', 'status', 'leader', 'support', 'disabled', 'hired', 'sherpa',
	'msuccess']].dropna(how='any', subset=['expid', 'myear', 'fname', 'lname', 'yob'])

climber_df.expid = climber_df.expid.str.cat(climber_df.myear)
climber_df['age'] = climber_df.myear.astype(int) - climber_df.yob.astype(int)

climber_df.sort_values(['fname', 'lname', 'yob', 'myear', 'mseason'], inplace=True)

climber_df['experience'] = climber_df.groupby(['fname', 'lname', 'yob'], as_index=False).expid.cumcount()
climber_df['leadership_experience'] = climber_df.groupby(['fname', 'lname', 'yob'], as_index=False).leader.cumsum()
climber_df['successes'] = climber_df.groupby(['fname', 'lname', 'yob'], as_index=False).msuccess.cumsum()

climber_df.experience = climber_df.experience.map(lambda x: max(x-1, 0))
climber_df.leadership_experience = climber_df.leadership_experience.map(lambda x: max(x-1, 0))
climber_df.successes = climber_df.successes.map(lambda x: max(x-1, 0))

team_df = climber_df.groupby('expid', as_index=False).agg({
	'experience': 'sum',
	'leadership_experience': 'sum',
	'successes': 'sum',
	'age': 'mean',
	'leader': 'sum',
	'support': 'sum',
	'disabled': 'sum',
	'hired': 'sum',
	'sherpa': 'sum'
})

team_df.rename({
	'experience': 'total_experience',
	'leadership_experience': 'total_leadership_experience',
	'successes': 'total_successes',
	'leader': 'leaders',
	'sherpa': 'sherpas',
	'age': 'avg_climber_age'
}, axis=1, inplace=True)

exped_cols = set(exped_df.columns)
team_cols = set(team_df.columns)

df = exped_df.merge(team_df, how='inner')
assert df.expid.nunique() == df.shape[0]

df['failure'] = 1 - df.success
df = df.drop(['expid', 'success'], axis=1).reset_index(drop=True)

# train test split
df_full_train, _, y_full_train, _ = train_test_split(
	df.drop('failure', axis=1), df.failure, test_size=0.2, random_state=1)


def one_hot_encoding(df, enc=None):
	df_categorical = df.select_dtypes(exclude='number').reset_index(drop=True)
	df_numerical = df.select_dtypes('number').reset_index(drop=True)

	if not enc:
		enc = OneHotEncoder(sparse_output=False, handle_unknown='infrequent_if_exist', drop='if_binary', dtype=np.int32)
		enc.fit(df_categorical)

	df_encoded = pd.DataFrame(data=enc.transform(df_categorical))

	X = pd.concat([df_encoded, df_numerical], axis=1).values

	return X, enc


X_train, enc = one_hot_encoding(df_full_train)
hyper_parameters = {'criterion': 'gini', 'max_depth': 20, 'min_samples_leaf': 15}
model = DecisionTreeClassifier(**hyper_parameters)
model.fit(X_train, y_full_train)