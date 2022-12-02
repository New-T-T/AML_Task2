import plotly.io as pio
import plotly.graph_objects as go
pio.renderers.default = 'browser'
import biosppy.signals.ecg as ecg
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots

# Load datasets, decompressing pickle with gzip
X_train = pd.read_pickle('data/X_train.pkl', compression='gzip')
y_train = pd.read_pickle('data/y_train.pkl', compression='gzip')
X_test = pd.read_pickle('data/X_test.pkl', compression='gzip')


# %%
X_train.set_index('id', inplace=True)
labels = pd.read_pickle('data/y_train.pkl', compression='gzip')
#labels = labels['label'].values

labels_array = labels['y'].to_numpy()
num_classes = 4

# Recover the id of one sample of each class (1st of each class)
sample_signal_ids = []
for class_id in range(num_classes):
    sample_signal_ids.append(int(np.argwhere(labels_array == class_id)[0]))

# Print indices of the sample signals
dic_samples = zip(labels['y'].unique(), sample_signal_ids)
print(f'Indices of the sample signals: {list(dic_samples)}')

#%% Plot 4 subplots, for 1 sample of each class

seconds = np.arange(0, 600) / 30
x_labels = [0, 5, 10, 15, 20]

fig = make_subplots(rows=4, cols=1, vertical_spacing=0.01, shared_xaxes=True)
for class_id in range(num_classes):
    print(len(X_train.loc[sample_signal_ids[class_id]].dropna().to_numpy(dtype='float32')))
    measurements = X_train.loc[sample_signal_ids[class_id]].dropna().to_numpy(dtype='float32')
    measurements = measurements[1000:7000:10]
    measurements /= 1000
    fig.add_trace(go.Scatter(x=seconds, y=measurements, mode='lines'), row=class_id+1, col=1)
    fig.update_yaxes(title_text=f"C{class_id} Amplitude (mV)", col=1, row=class_id+1)
    fig.update_xaxes(tickvals=x_labels, col=1, row=class_id+1)

fig.update_layout(title_text="Sample signals")
fig.update_xaxes(title_text="Time (s)", col=1, row=4)
# setting x_ticks
fig.show()

#%% Plot 4 subplots, the ECG with the R-peaks, for 1 sample of each class

signals = X_train.iloc[sample_signal_ids]
for class_id in range(num_classes):
    measurements = signals.iloc[class_id].dropna().to_numpy(dtype='float32')
    r_peaks = ecg.engzee_segmenter(signals.iloc[class_id], 300)['rpeaks']
    r_peaks_in_window = []
    j = 0
    h = 0
    for i in range(len(measurements)):
        if r_peaks[j] - 5 < i < r_peaks[j] + 6:
            r_peaks_in_window.append(1)
            h = h+1
            print(f'j = {j}, h = {h}')
            if h == 10:
                j += 1
                h = 0
                if j == len(r_peaks):
                    break
        else:
            r_peaks_in_window.append(0)
    measurements = measurements[1000:7000:10]
    measurements /= 1000
    r_peaks_in_window = r_peaks_in_window[1000:7000:10]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=seconds, y=measurements, mode='lines'))
    for id_peak, r_peak in enumerate(r_peaks_in_window):
        if r_peak == 1:
            fig.add_vline(x=id_peak/30, line_width=1, line_dash="dash", line_color="red")
    fig.show()

#%% Plot 4 subplots, the heartbeat, for 1 sample of each class

colors = ['blue', 'red', 'green', 'violet']
fig_signals = make_subplots(rows=4, cols=2, vertical_spacing=0.01, horizontal_spacing=0.03, shared_xaxes=True)
fig_heartbeat = make_subplots(rows=1, cols=4, horizontal_spacing=0.01, shared_yaxes=True)
for class_id in range(num_classes):
    r_peaks = ecg.engzee_segmenter(signals.iloc[class_id], 300)['rpeaks']
    if len(r_peaks) >= 2:
        # print(ecg.extract_heartbeats(signals[0], r_peaks, 300))
        beats = ecg.extract_heartbeats(signals.iloc[class_id].dropna().to_numpy(dtype='float32'), r_peaks, 300)['templates']
        print(beats.shape)
        if len(beats) != 0:
            mu = np.mean(beats, axis=0)
            var = np.std(beats, axis=0)
            md = np.median(beats, axis=0)
    mu = mu / 1000
    var = var / 1000
    md = md / 1000
    if class_id == 0:
        md_0 = md
    x_axis = np.array(range(len(mu)))
    fig_heartbeat.add_trace(go.Scatter(x=x_axis, y=mu, mode='lines', name='mean', line=dict(color='red')), row=1, col=class_id+1)
    mu_upper = mu + var
    mu_lower = mu - var
    std_plot = go.Scatter(
        x=np.append(x_axis, x_axis[::-1]),  # x, then x reversed
        y=np.append(mu_upper, mu_lower[::-1]),  # upper, then lower reversed
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        # line color grey
        line=dict(color='rgba(255,255,255,0)',
                  width=0),
        hoverinfo="skip",
        showlegend=False,
        mode='lines'
    )
    fig_heartbeat.add_trace(std_plot, row=1, col=class_id+1)
    fig_heartbeat.add_trace(go.Scatter(x=x_axis, y=md, mode='lines', name='median', line=dict(color='blue')), row=1, col=class_id+1)
    measurements = signals.iloc[class_id].dropna().to_numpy(dtype='float32')
    measurements = measurements[1000:7000:10]
    measurements /= 1000
    measurements_median = measurements / max(md)
    measurements_median0 = measurements / max(md_0)
    fig_signals.add_trace(go.Scatter(x=seconds, y=measurements_median, mode='lines', line=dict(color=colors[class_id])), row=class_id+1, col=1)
    fig_signals.add_trace(go.Scatter(x=seconds, y=measurements_median0, mode='lines', line=dict(color=colors[class_id])), row=class_id+1, col=2)
    fig_signals.update_yaxes(range=[-2, 2], row=class_id+1)

fig_heartbeat.show()
fig_signals.update_xaxes(range=[0, 10], col=1, row=4)
fig_signals.update_xaxes(range=[0, 10], col=2, row=4)
fig_signals.show()

#%%
import neurokit2 as nk
X_train_healthy = X_train[y_train['y'] == 0].copy()
nb_cols = 10
nb_rows = 8

fig_heartbeat = make_subplots(rows=nb_rows, cols=nb_cols, shared_xaxes=True, vertical_spacing=0.01)
median_amplitudes_healthy = []
for id_loop, row_id in enumerate(X_train_healthy.index[:nb_rows*nb_cols]):
    print(row_id)
    signal = X_train_healthy.loc[row_id].dropna().to_numpy(dtype='float32')
    signal, is_inverted = nk.ecg_invert(signal, sampling_rate=300)
    if is_inverted:
        print('inverted')
    signal = nk.ecg_clean(signal, sampling_rate=300, method='engzeemod2012')
    r_peaks = ecg.engzee_segmenter(signal, 300)['rpeaks']
    if len(r_peaks) >= 2:
        # print(ecg.extract_heartbeats(signals[0], r_peaks, 300))
        beats = ecg.extract_heartbeats(signal, r_peaks, 300)['templates']
        if len(beats) != 0:
            mu = np.mean(beats, axis=0)
            var = np.std(beats, axis=0)
            md = np.median(beats, axis=0)

    mu = mu / 1000
    var = var / 1000
    md = md / 1000
    x_axis = np.array(range(len(mu)))
    if id_loop < 5:
        fig_heartbeat.add_trace(go.Scatter(x=x_axis, y=mu, mode='lines', name='mean', line=dict(color='red')), row=1, col=id_loop+1)
    mu_upper = mu + var
    mu_lower = mu - var
    std_plot = go.Scatter(
        x=np.append(x_axis, x_axis[::-1]),  # x, then x reversed
        y=np.append(mu_upper, mu_lower[::-1]),  # upper, then lower reversed
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        # line color grey
        line=dict(color='rgba(255,255,255,0)',
                  width=0),
        hoverinfo="skip",
        showlegend=False,
        mode='lines'
    )
    fig_heartbeat.add_trace(std_plot, row=int(id_loop/nb_cols + 1), col=id_loop%nb_cols + 1)
    fig_heartbeat.add_trace(go.Scatter(x=x_axis, y=md, mode='lines', name='median', line=dict(color='blue')), row=int(id_loop/nb_cols + 1), col=id_loop%nb_cols + 1)

fig_heartbeat.show()

#%%
seconds = np.arange(0, 600) / 30
x_labels = [0, 5, 10, 15, 20]

fig = make_subplots(rows=6, cols=1, vertical_spacing=0.01, shared_xaxes=True)
for id_loop, row_id in enumerate([33, 35, 46, 47, 48, 72]):
    print(X_train_healthy.iloc[row_id].name)
    print(len(X_train_healthy.iloc[row_id].dropna().to_numpy(dtype='float32')))
    measurements = X_train_healthy.iloc[row_id].dropna().to_numpy(dtype='float32')
    measurements = measurements[1000:7000:10]
    measurements /= 1000
    fig.add_trace(go.Scatter(x=seconds, y=measurements, mode='lines'), col=1, row=id_loop+1)
    fig.update_yaxes(title_text=f"{row_id} Amplitude (mV)", col=1, row=id_loop+1)
    fig.update_xaxes(tickvals=x_labels, col=1, row=id_loop+1)

fig.update_layout(title_text="Sample signals")
fig.update_xaxes(title_text="Time (s)", col=1, row=4)
# setting x_ticks
fig.show()
#%%
import pandas as pd
import plotly.express as px
import neurokit2 as nk

import matplotlib.pyplot as plt

ecg = nk.ecg_simulate(duration=10, sampling_rate=1000)

signals = pd.DataFrame({"ECG_Raw" : ecg,
                        "ECG_NeuroKit" : nk.ecg_clean(ecg, sampling_rate=1000, method="neurokit"),
                        "ECG_BioSPPy" : nk.ecg_clean(ecg, sampling_rate=1000, method="biosppy"),
                        "ECG_PanTompkins" : nk.ecg_clean(ecg, sampling_rate=1000, method="pantompkins1985"),
                        "ECG_Hamilton" : nk.ecg_clean(ecg, sampling_rate=1000, method="hamilton2002"),
                        "ECG_Elgendi" : nk.ecg_clean(ecg, sampling_rate=1000, method="elgendi2010"),
                        "ECG_EngZeeMod" : nk.ecg_clean(ecg, sampling_rate=1000, method="engzeemod2012"),
                        "ECG_standardize" : nk.standardize(ecg, sampling_rate=1000)})

# Plot
fig = px.line(signals, x=signals.index, y=signals.columns)
fig.show()

