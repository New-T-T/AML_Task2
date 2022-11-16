import pandas as pd

X_train = pd.read_csv('data/X_train.csv')
y_train = pd.read_csv('data/y_train.csv')
X_test = pd.read_csv('data/X_test.csv')

# Save these datasets in pickle with gzip
X_train.to_pickle('data/X_train.pkl', compression='gzip')
y_train.to_pickle('data/y_train.pkl', compression='gzip')
X_test.to_pickle('data/X_test.pkl', compression='gzip')