import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from biosppy.signals import ecg
import neurokit2 as nk
from imblearn.over_sampling import BorderlineSMOTE


def get_data():
    X_train = pd.read_pickle('data/X_train.pkl', compression='gzip')
    y_train = pd.read_pickle('data/y_train.pkl', compression='gzip')
    X_test = pd.read_pickle('data/X_test.pkl', compression='gzip')
    X_train.set_index('id', inplace=True)
    X_test.set_index('id', inplace=True)
    y_train.set_index('id', inplace=True)
    return X_train, y_train, X_test

def cleaning_and_zero_padding(X_train, X_test):
    # Zero padding
    # erase the upper triangle of the matrix
    list_signals = []
    nb_inverted = [0,0]
    max_len = X_train.shape[1]
    print(f'max_len: {max_len}')
    for row in X_train.index:
        print(f'row: {row}')
        # Getting the signal and dropping the NaN values
        signal = X_train.loc[row].dropna().to_numpy(dtype='float32')
        signal, is_inverted = nk.ecg_invert(signal, sampling_rate=300)
        if is_inverted:
            nb_inverted[0] += 1
        # zero padding before and after the signal
        if (max_len - len(signal)) % 2 == 0:
            pad = int((max_len - len(signal)) / 2)
            signal = np.pad(signal, (pad, pad), 'constant')
        else:
            pad = int((max_len - len(signal)) / 2)
            signal = np.pad(signal, (pad, pad + 1), 'constant')
        list_signals.append(signal)
    X_train_pad = pd.DataFrame(list_signals, index=X_train.index)

    list_signals = []
    for row in X_test.index:
        print(f'row: {row}')
        # Getting the signal and dropping the NaN values
        signal = X_test.loc[row].dropna().to_numpy(dtype='float32')
        signal, is_inverted = nk.ecg_invert(signal, sampling_rate=300)
        if is_inverted:
            nb_inverted[1] += 1
        # zero padding before and after the signal
        if (max_len - len(signal)) % 2 == 0:
            pad = int((max_len - len(signal)) / 2)
            signal = np.pad(signal, (pad, pad), 'constant')
        else:
            pad = int((max_len - len(signal)) / 2)
            signal = np.pad(signal, (pad, pad + 1), 'constant')
        list_signals.append(signal)
    X_test_pad = pd.DataFrame(list_signals, index=X_test.index)
    print(f'nb_inverted: {nb_inverted}')
    return X_train_pad, X_test_pad

def preprocess_data(X_train, y_train, X_test, seed):
    # Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train,
                                                      random_state=seed)

def balance_classes(X_train, y_train):
    #smote = SMOTE()
    smote = BorderlineSMOTE(sampling_strategy = 'not majority')
    # fit predictor and target variable
    x_smote, y_smote = smote.fit_resample(X_train, y_train)

    #r_under_sampler = RandomUnderSampler(random_state=42, replacement=True) # fit predictor and target variable
    #x_undersampled, y_undersampled = rus.fit_resample(X_train, y_train)

    return x_smote, y_smote

#%%
seed = 42
verbose = 3
X_train, y_train, X_test = get_data()
X_train_pad, X_test_pad = cleaning_and_zero_padding(X_train, X_test)
#%%
X_train_pad_orig = X_train_pad.copy()
#%%
X_train_pad, X_val_pad, y_train_pad, y_val_pad = train_test_split(X_train_pad, y_train, test_size=0.2, stratify=y_train, random_state=seed)
#%%
# set the name of the index column to 'id'
X_train_pad.index.name = 'id'
X_val_pad.index.name = 'id'
X_test_pad.index.name = 'id'
#%%
# Save the data in csv files in data/train_val_split_pad folder
X_train_pad.to_csv('data/train_val_split_pad/X_train_pad.csv')
X_val_pad.to_csv('data/train_val_split_pad/X_val_pad.csv')
y_train_pad.to_csv('data/train_val_split_pad/y_train_pad.csv')
y_val_pad.to_csv('data/train_val_split_pad/y_val_pad.csv')
X_test_pad.to_csv('data/train_val_split_pad/X_test_pad.csv')
#%%
X_test_pad3 = pd.read_csv('data/train_val_split_pad/X_test_pad.csv', index_col='id')
#%%
y_train_pad2 = pd.read_csv('data/train_val_split_pad/y_train_pad.csv')
#%%

import pandas as pd
X_train_pad = pd.read_csv('data/original_pad/X_train_pad.csv', index_col='id')
#%%
y_train_pad = pd.read_csv('data/original_pad/y_train_pad.csv', index_col='id')







#%%
import pandas as pd
X_train_pad = pd.read_pickle('data/original_pad/X_train_pad.pkl', compression='gzip')
X_test_pad = pd.read_pickle('data/original_pad/X_test_pad.pkl', compression='gzip')
y_train = pd.read_pickle('data/y_train.pkl', compression='gzip')
#%%
# set id as index for y
y_train.set_index('id', inplace=True)
#%%
X_train_pad_balanced, y_train_balanced = balance_classes(X_train_pad, y_train)
#%%
X_train_pad_csv = pd.read_csv('data/original_pad/X_train_pad.csv', index_col='id')
#%%
# import pandas as pd
# import numpy as np
#
# df = pd.DataFrame(np.random.randn(10, 6), columns=['a', 'b', 'c', 'd', 'e', 'g'])
# # erase the upper triangle of the matrix
# df = df.where(np.tril(np.ones(df.shape)).astype(np.bool))
# list_values = []
# for row in df.index:
#     print(f'row: {row}')
#     row = df.loc[row]
#     values = row.values
#     # drop the nan values
#     values = values[~np.isnan(values)]
#     # zero padding before and after the signal
#     print(f'values: {values}, len: {len(values)}')
#     print(f'pad = {((df.shape[1]-len(values))/2)}')
#     if (df.shape[1]-len(values))%2 == 0:
#         pad = int((df.shape[1]-len(values))/2)
#         values = np.pad(values, (pad, pad), 'constant')
#         print(f'values: {values}, len: {len(values)}')
#     else:
#         pad = int((df.shape[1]-len(values))/2)
#         values = np.pad(values, (pad, pad+1), 'constant')
#         print(f'values: {values}, len: {len(values)}')
#     list_values.append(values)
# df_pad = pd.DataFrame(list_values, index=df.index)

# df = pd.DataFrame(np.random.randn(10, 6), columns=['a', 'b', 'c', 'd', 'e', 'g'])
# df_test = pd.DataFrame(np.random.randn(10, 6), columns=['a', 'b', 'c', 'd', 'e', 'g'])
# # erase the upper triangle of the matrix
# df = df.where(np.tril(np.ones(df.shape)).astype(np.bool))
# df_test = df_test.where(np.tril(np.ones(df_test.shape)).astype(np.bool))
# df_pad, df_test_pad = cleaning_and_zero_padding(df, df_test)
