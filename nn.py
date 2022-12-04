import numpy as np
import pandas as pd
import torch
from torch import nn

# Setting the seed for generating random numbers ensure that reproducible results
SEED = 42
torch.manual_seed(SEED)

print(torch.cuda.is_available())
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Google colab offers time limited use of GPU for free
print(device)

#%%
from cnn1d.data_loading import get_datasets, get_dataloaders
PHASE = 'testing'
if PHASE == 'training':
    # train_ds, predict_ds, y_train, y_predict = get_datasets(device=device, phase='training')
    train_ds, val_ds, predict_ds = get_datasets(device=device, phase='training')

elif PHASE == 'testing':
    train_ds, val_ds, predict_ds = get_datasets(device=device, phase='testing')
    # train_ds, predict_ds, y_train = get_datasets(device=device, phase='testing')

#%%
BATCH_SIZE = 30
X_train_dl, X_val_dl, X_predict_dl, class_weights = get_dataloaders(train_ds=train_ds, val_ds=val_ds, # y_train=y_train,
                                                                    predict_ds=predict_ds,
                                                                    batch_size=BATCH_SIZE, seed=SEED)
#%%
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.renderers.default = "browser"

y_all = []
for X_batch, y_batch in X_val_dl:
    print(X_batch.shape)
    print(y_batch.shape)
    y_array = y_batch.numpy(force=True).tolist()
    y_all.append(y_array)
# display the distribution of the classes for each batch
fig = go.Figure()
for i in range(len(y_all)):
    y = y_all[i]
    print(y)
    fig.add_trace(go.Histogram(x=y, name=f'Batch {i}', histnorm='probability density'))
fig.show()
#%%
from cnn1d.model import MLP
from cnn1d.training import train

# model definition
FEATURES_NB = 17807
model = MLP(n_outputs=4)
model.to(device)

# import torchsummary as ts
# ts.summary(model, input_size=(BATCH_SIZE, 1, FEATURES_NB))
#%%
# Training
# criterion = nn.BCEWithLogitsLoss()
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
model, loss_stats, accuracy_stats, precision_stats, recall_stats, f1_stats = train(model, X_train_dl, X_val_dl, 10, criterion, device)
#%%
from cnn1d.testing import predict
# Prediction
best_model = MLP(FEATURES_NB)
best_model.load_state_dict(torch.load('model.path', map_location=device))
best_model.to(device)
best_model.eval()
predicted = pd.DataFrame(predict(best_model, X_predict_dl, device))
labels_predicted = predicted.idxmax(axis=1)
#%%
from cnn1d.data_loading import ds_to_df
signals, labels = ds_to_df(ecg_predict_ds)
#%%
# import f1 score from sklearn
from sklearn.metrics import f1_score
# compute f1 score for the predicted labels
f1_score(labels, labels_predicted, average='micro')
#%%
predicted_submission = predict(best_model, X_test_predict_dl, device)
predicted_submission_df = pd.DataFrame(np.concatenate(predicted_submission))
predicted_submission_df
#%%
labels_submission = predicted_submission_df.idxmax(axis=1)
labels_submission.columns = ['y']