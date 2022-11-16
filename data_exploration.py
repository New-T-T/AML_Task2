import pandas as pd

# Load datasets, decompressing pickle with gzip
X_train = pd.read_pickle('data/X_train.pkl', compression='gzip')
y_train = pd.read_pickle('data/y_train.pkl', compression='gzip')
X_test = pd.read_pickle('data/X_test.pkl', compression='gzip')
