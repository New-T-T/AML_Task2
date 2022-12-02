import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from sklearn.preprocessing import OneHotEncoder


PATH = ['/content/drive/MyDrive/task2', 'data/split_pad']


class ECGDataset(Dataset):
    # load the dataset
    def __init__(self, dataset_name: str, id_path):
        # set id_path to 0 if device is cpu
        # set id_path to 1 if device is cuda
        self.dataset_name = dataset_name
        self.X = pd.read_csv(f'{PATH[id_path]}/X_{dataset_name}_pad.csv', index_col='id')
        self.X = torch.from_numpy(self.X.to_numpy()).float()
        # self.X.to(device)
        if dataset_name in ['train', 'trainval', 'predict']:
            self.y = pd.read_csv(f'{PATH[id_path]}/y_{dataset_name}_pad.csv', index_col='id')
            self.y = torch.tensor(self.y['y'].values, dtype=torch.long)
            # self.y = torch.nn.functional.one_hot(self.y, num_classes=4)
            # self.y.to(device)

    # returns the number of instances in the dataset
    def __len__(self):
        return len(self.X)

    # returns one instance {x, y}
    def __getitem__(self, id):
        if self.dataset_name == 'test':
            return self.X[id, :]
        else:
            return self.X[id, :], self.y[id]

    def get_X(self):
        return self.X

def get_datasets(device, phase):
    id_path = 0 if device != 'cpu' else 1
    if phase == 'training':
        ecg_ds = ECGDataset(dataset_name='trainval', id_path=id_path)
        y_train = pd.read_csv(f'{PATH[id_path]}/y_trainval_pad.csv', index_col='id')
        ecg_predict_ds = ECGDataset(dataset_name='predict', id_path=id_path)
        y_predict = pd.read_csv(f'{PATH[id_path]}/y_predict_pad.csv', index_col='id')
        return ecg_ds, ecg_predict_ds, y_train, y_predict
    if phase == 'testing':
        ecg_ds = ECGDataset(dataset_name='train', id_path=id_path)
        y_train = pd.read_csv(f'{PATH[id_path]}/y_train_pad.csv', index_col='id')
        ecg_predict_ds = ECGDataset(dataset_name='test', id_path=id_path)
        return ecg_ds, ecg_predict_ds, y_train


def get_dataloaders(train_ds, y_train, predict_ds, batch_size=1, train_split=0.85, val_split=0.15, seed=42):
    # Splitting the data in Train-Val-Test
    nb_training_instances = int(train_split * len(train_ds))  # length of the training set
    # nb_validation_instances = int(val_split * len(ecg_ds))  # length of the validation set
    # nb_testing_instances = len(ecg_ds) - (nb_training_instances + nb_validation_instances)
    nb_validation_instances = len(train_ds) - nb_training_instances
    len_ds = [nb_training_instances, nb_validation_instances] #, nb_testing_instances]

    print(f'ECG full dataset size: {len(train_ds)}')
    print(f'ECG training dataset size: {nb_training_instances}')
    print(f'ECG validation dataset size: {nb_validation_instances}')
    # print(f'ECG testing dataset size: {nb_testing_instances}\n')

    # Random split
    # X_train_ds, X_val_ds, X_test_ds = random_split(dataset=ecg_ds, lengths=len_ds, \
    #                                                generator=torch.Generator().manual_seed(seed))
    X_train_ds, X_val_ds = random_split(dataset=train_ds, lengths=len_ds, \
                                        generator=torch.Generator().manual_seed(seed))

    # Class weights
    weights = 1 / y_train.value_counts(normalize=False, sort=False)
    class_weights = torch.tensor(weights.values, dtype=torch.float)
    class_weights_all = class_weights[y_train.values]
    class_weights_all = class_weights_all.reshape(-1)
    weighted_sampler = WeightedRandomSampler(
        weights=class_weights_all,
        num_samples=len(class_weights_all),
        replacement=True
    )
    # Data loader
    X_train_dl = DataLoader(X_train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    X_val_dl = DataLoader(X_val_ds, batch_size=batch_size)
    # X_test_dl = DataLoader(X_test_ds, batch_size=batch_size)
    X_predict_dl = DataLoader(predict_ds, batch_size=batch_size)
    return X_train_dl, X_val_dl, X_predict_dl, class_weights #, X_test_dl

# convert ds to dataframe
def ds_to_df(ds):
    signals = []
    labels = []
    for i in range(len(ds)):
        signals.append(ds[i][0].numpy())
        labels.append(ds[i][1][0].numpy())
    signals = pd.DataFrame(signals)
    labels = pd.DataFrame(labels)
    enc = OneHotEncoder()
    labels_to_fit = [[0], [1], [2], [3]]
    encoder = enc.fit(labels_to_fit)
    labels = pd.DataFrame(encoder.inverse_transform(labels), columns=['y'])
    return signals, labels