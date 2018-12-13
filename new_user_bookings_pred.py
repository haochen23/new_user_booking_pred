import pandas as pd
import numpy as np

# Import libraries
import xgboost as xgb
from sklearn import decomposition
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.preprocessing import LabelEncoder
#import data
tr_datapath = "data/train_users_2.csv"
te_datapath = "data/test_users.csv"
df_train = pd.read_csv(tr_datapath, header=0, index_col=None)
df_test = pd.read_csv(te_datapath, header=0, index_col=None)

# combine df_train and df_test into one DataFrame
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True, sort=False)
# fixing the date_account_created column
df_all['date_account_created'] = pd.to_datetime(df_all['date_account_created'], format='%Y-%m-%d')
# fixing the timestamp_first_active column
df_all['timestamp_first_active'] = pd.to_datetime(df_all['timestamp_first_active'], format='%Y%m%d%H%M%S')
# use the timestamp_first_active column to fill the missing values in data_account_created column
df_all['date_account_created'].fillna(df_all.timestamp_first_active, inplace=True)

# Drop the date_first_booking column
df_all.drop('date_first_booking', axis=1, inplace=True)

# avoid comparison with NaN values
df_all['age'].fillna(-1, inplace=True)

# function to clean incorrect value


def remove_outliers(df, column, min_val, max_val):
    col_values = df[column].values
    df[column] = np.where(np.logical_or(col_values < min_val, col_values > max_val), np.NaN, col_values)
    return df


# Fixing age column
df_all = remove_outliers(df=df_all, column='age', min_val=15, max_val=90)
df_all['age'].fillna(-1, inplace=True)
# Fill missing values in first_affliate_tracked
df_all['first_affiliate_tracked'].fillna(-1, inplace=True)


# one hot encoding function
def convert_to_onehot(df, column_to_convert):
    categories = list(df[column_to_convert].drop_duplicates())

    for category in categories:
        cat_name = str(category).replace(" ", "_").replace(
            "(", "").replace(")", "").replace("/", "_").replace("-", "").lower()
        col_name = column_to_convert[:5] + '_' + cat_name[:10]
        df[col_name] = 0
        df.loc[(df[column_to_convert] == category), col_name] = 1
    return df


# One hot encoding, and drop the original column from df_all
columns_to_convert = ['gender', 'signup_method', 'signup_flow', 'language',
                      'affiliate_channel', 'affiliate_provider',
                      'first_affiliate_tracked', 'signup_app',
                      'first_device_type', 'first_browser']

for column in columns_to_convert:
    df_all = convert_to_onehot(df_all, column)
    df_all.drop(column, axis=1, inplace=True)

# Add new datetime related fields
df_all['day_account_created'] = df_all['date_account_created'].dt.weekday
df_all['month_account_created'] = df_all['date_account_created'].dt.month
df_all['quarter_account_created'] = df_all['date_account_created'].dt.quarter
df_all['year_account_created'] = df_all['date_account_created'].dt.year
df_all['hour_first_active'] = df_all['timestamp_first_active'].dt.hour
df_all['day_first_active'] = df_all['timestamp_first_active'].dt.weekday
df_all['month_first_active'] = df_all['timestamp_first_active'].dt.month
df_all['quarter_first_active'] = df_all['timestamp_first_active'].dt.quarter
df_all['year_first_active'] = df_all['timestamp_first_active'].dt.year
df_all['created_less_active'] = (df_all['date_account_created'] - df_all['timestamp_first_active']).dt.days

# Drop unnecessary columns
columns_to_drop = ['date_account_created', 'timestamp_first_active', 'date_first_booking', 'country_destination']
for column in columns_to_drop:
    if column in df_all.columns:
        df_all.drop(column, axis=1, inplace=True)

# read sessions.csv
session_path = 'data/sessions.csv'
sessions = pd.read_csv(session_path, header=0, index_col=False)
# Determine primary device
sessions_device = sessions.loc[:, ['user_id', 'device_type', 'secs_elapsed']]
aggregated_lvl1 = sessions_device.groupby(['user_id', 'device_type'],
                                          as_index=False, sort=False).aggregate(np.sum)
index = aggregated_lvl1.groupby(['user_id'], sort=False)[
    'secs_elapsed'].transform(max) == aggregated_lvl1['secs_elapsed']
df_primary = pd.DataFrame(aggregated_lvl1.loc[
    index, ['user_id', 'device_type', 'secs_elapsed']])
df_primary.rename(columns={
    'device_type': 'primary_device',
    'secs_elapsed': 'primary_secs'}, inplace=True)
df_primary = convert_to_onehot(df_primary, column_to_convert='primary_device')
df_primary.drop('primary_device', axis=1, inplace=True)

# Determine secondary device
remaining = aggregated_lvl1.drop(aggregated_lvl1.index[index])
index = remaining.groupby(
    ['user_id'], sort=False)['secs_elapsed'].transform(max) == remaining['secs_elapsed']
df_secondary = pd.DataFrame(
    remaining.loc[index, ['user_id', 'device_type', 'secs_elapsed']])
df_secondary.rename(columns={
    'device_type': 'secondary_device', 'secs_elapsed': 'secondary secs'}, inplace=True)
df_secondary = convert_to_onehot(df_secondary, 'secondary_device')
df_secondary.drop('secondary_device', axis=1, inplace=True)

# function to count occurrences of value in a column


def convert_to_counts(df, id_col, column_to_convert):
    id_list = df[id_col].drop_duplicates()
    df_counts = df.loc[:, [id_col, column_to_convert]]
    df_counts['count'] = 1
    df_counts = df_counts.groupby(by=[id_col, column_to_convert],
                                  as_index=False, sort=False).sum()

    new_df = df_counts.pivot(index=id_col, columns=column_to_convert, values='count')
    new_df = new_df.fillna(0)

    # rename columns
    categories = list(df[column_to_convert].drop_duplicates())
    for category in categories:
        cat_name = str(category).replace(
            " ", "_").replace("(", "").replace(")", "").replace(
            "/", "_").replace("-", "").lower()
        col_name = column_to_convert + '_' + cat_name
        new_df.rename(columns={category: col_name}, inplace=True)

    return new_df


# Aggregate and combine actions taken columns
session_actions = sessions.loc[:, ['user_id', 'action', 'action_type', 'action_detail']]
columns_to_convert = ['action', 'action_type', 'action_detail']
session_actions = session_actions.fillna('not provided')

# flag indicating the first loop
first = True

for column in columns_to_convert:
    print("Converting " + column + " column...")
    current_data = convert_to_counts(df=session_actions, id_col='user_id', column_to_convert=column)
    if first:
        first = False
        actions_data = current_data
    else:
        actions_data = pd.concat([actions_data, current_data], axis=1, join='inner')

# Combine device datasets
df_primary.set_index('user_id', inplace=True)
df_secondary.set_index('user_id', inplace=True)
device_data = pd.concat([df_primary, df_secondary], axis=1, join='outer', sort=False)

# Combine device and actions datasets
combined_results = pd.concat([device_data, actions_data], axis=1, join='outer', sort=False)
df_sessions = combined_results.fillna(0)

# Combine user and sessions datasets
df_all.set_index('id', inplace=True)
df_all = pd.concat([df_all, df_sessions], axis=1, join='inner', sort=False)
# Prepare training data for model training
df_train.set_index('id', inplace=True)
df_train = pd.concat([df_train['country_destination'], df_all], axis=1, join='inner', sort=False)

index_train = df_train.index.values
labels = df_train['country_destination']
le = LabelEncoder()
y = le.fit_transform(labels)  # training labels
x = df_train.drop('country_destination', axis=1, inplace=False)  # training data

# Grid Search - used to find the best combination of parameters
model = xgb.XGBClassifier(learning_rate=0.1, max_depth=5, n_estimators=25, objective='multi:softprob'
                          )
param_grid = {'max_depth': [5],
              'learning_rate': [0.1], 'n_estimators': [25]}
# model = GridSearchCV(estimator=XGB_model, param_grid=param_grid,
#                      scoring='accuracy', verbose=10, n_jobs=1,
#                      iid=True, refit=True, cv=3)
print("so far so good")
# Model training
model.fit(x, y)
print("Best score: %0.3f" % model.best_score_)
print("Best parameters set:")
best_parameters = model.best_estimator_.get_params()
for param_name in sorted(param_grid.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

# Prepare test data for prediction
df_test.set_index('id', inplace=True)
df_test = pd.merge(df_test.loc[:, ['date_first_booking']],
                   df_all, how='left', left_index=True,
                   right_index=True, sort=False)
x_test = df_test.drop('date_first_booking', axis=1, inplace=False)
x_test = x_test.fillna(-1)
index_test = df_test.index.values

# Make predictions
y_pred = model.predict_proba(x_test)
# Taking the 5 classes with highest probabilities
ids = []  # list of ids
cts = []  # list of countries
for i in range(len(id_test)):
    idx = id_test[i]
    ids += [idx] * 5
    cts += le.inverse_transform(np.argsort(y_pred[i])[::-1])[:5].tolist()

# Generate submission
print("Outputting final results...")
sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
sub.to_csv('./submission.csv', index=False)
