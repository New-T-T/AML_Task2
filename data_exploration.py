import pandas as pd

# Feature Extraction
import biosppy.signals.ecg as ecg
import neurokit2 as nk
import heartpy as hp

import numpy as np

import scipy.stats as stats

# Load datasets, decompressing pickle with gzip
X_train = pd.read_pickle('data/X_train.pkl', compression='gzip')
y_train = pd.read_pickle('data/y_train.pkl', compression='gzip')
X_test = pd.read_pickle('data/X_test.pkl', compression='gzip')

# Feature Selection

# R_peaks variance
    #Feature 1: number of spikes/total peaks
    #Feature 2: STD (after modified zscore)
    #Feature 3: MAD (after modified zscore)
    #Feature 4: peak median
    #Feature 5: peak mean
    #Feature 6: min peak
    #Feature 7: max peak

# modified zscore based on MAD (median absolute deviation)
def modified_zscore(data, consistency_correction = 1.4826):
    median = np.median(data)
    deviation_from_med = np.array(data) - median

    mad = np.median(np.abs(deviation_from_med))
    mod_zscore = deviation_from_med/(consistency_correction*mad)

    return mod_zscore

def feat_rpeak_variance(data_ts, static_rpeak_threshold = 400):           #returns an array of the features

    feature_rpeaks_variance = np.empty(7)

    for i in range(len(data_ts)):

        #capture the signal (clean and use neurokit.ecg_findpeaks)
        raw_signal = data_ts[i,][~np.isnan(data_ts[i,])]
        cleaned_signal = nk.ecg_clean(raw_signal, sampling_rate = 300, method='engzeemod2012')
        r_peaks_cleaned = nk.ecg_findpeaks(cleaned_signal, 300)['ECG_R_Peaks']

        #extract the values of the peaks
        rpeak_values_cleaned = cleaned_signal[r_peaks_cleaned]

        #median & mean
        median_rpeaks = np.median(rpeak_values_cleaned)
        mean_rpeak_cleaned = np.mean(rpeak_values_cleaned)

        #max
        try:
            max_rpeaks = np.max(rpeak_values_cleaned)

        except ValueError:
            max_rpeaks = np.nan

        #min
        try:
            min_rpeaks = np.min(rpeak_values_cleaned)

        except ValueError:
            min_rpeaks = np.nan

        # (modified)-zscore for rpeak-outlier detection

        #zscore_cleaned = stats.zscore(rpeak_values_cleaned, axis = 0, nan_policy = 'omit') #array with zscore for each element
        modified_zscore_cleaned = modified_zscore(rpeak_values_cleaned)

        #z_threshold = 2
        modified_z_threshold = 10

        for t in range(len(rpeak_values_cleaned)):
            if abs(modified_zscore_cleaned[t]) > modified_z_threshold:        #replace the outliers
                rpeak_values_cleaned[t] = median_rpeaks                       #replace outlier peaks by the median

        #STD & MAD
        std_rpeaks = np.std(rpeak_values_cleaned)
        mad_rpeaks = stats.median_abs_deviation(rpeak_values_cleaned)

        #detect spikes (i.e., unusually high R_peaks)
        counter_spikes = 0
        for j in r_peaks_cleaned:
            if abs(cleaned_signal[j]) > abs((median_rpeaks + static_rpeak_threshold)):
                counter_spikes = counter_spikes + 1

        if len(r_peaks_cleaned) != 0:
            counter_spikes /= len(r_peaks_cleaned)                            #normalize the spikes with respect to the total number of R_peaks

        #add features
        var_feat = [counter_spikes, std_rpeaks, mad_rpeaks, median_rpeaks, median_rpeaks, max_rpeaks, min_rpeaks]
        feature_rpeaks_variance = np.vstack([feature_rpeaks_variance, var_feat])

    feature_rpeaks_variance = np.delete(feature_rpeaks_variance, 0, 0)        #delete the (random) first column

    print("R_peak variance feature calculated!")

    return feature_rpeaks_variance

#P,Q,S,T-peaks and cardiac phase
#24 features in total
#For each peak type:
    #Feature1: mean
    #Feature2: median
    #Feature3: max
    #Feature4: min
    #Feature5: STD
    #Feature6: MAD

def PQST_peaks(data_ts):

    feature_PQST_peaks = np.empty(24)

    counter = 0 # for counting the skipped iterations

    for i in range(len(data_ts)):

        print(f"Iteration {i}")

        try:
            #capture the signal and clean it
            raw_signal = data_ts[i,][~np.isnan(data_ts[i,])]

            # delineate the cleaned ECG signal to get the P,Q,S,T-peaks
            cleaned_signal = nk.ecg_clean(raw_signal, sampling_rate=300, method="neurokit")
            _, rpeaks = nk.ecg_peaks(cleaned_signal, sampling_rate=300)
            signals, waves = nk.ecg_delineate(cleaned_signal, rpeaks, sampling_rate=300, method="dwt")

            # Q-peak features
            qpeaks_mean = np.mean(waves["ECG_Q_Peaks"])
            qpeaks_median = np.median(waves["ECG_Q_Peaks"])
            qpeaks_max = np.max(waves["ECG_Q_Peaks"])
            qpeaks_min = np.min(waves["ECG_Q_Peaks"])
            qpeaks_std = np.std(waves["ECG_Q_Peaks"])
            qpeaks_mad = stats.median_abs_deviation(waves["ECG_Q_Peaks"])

        except:
            #print(f"No result for {i}.")
            counter += 1
            qpeaks_mean = np.nan
            qpeaks_median = np.nan
            qpeaks_max = np.nan
            qpeaks_min = np.nan
            qpeaks_std = np.nan
            qpeaks_mad = np.nan

        try:
            # P-peak features
            ppeaks_mean = np.mean(waves["ECG_P_Peaks"])
            ppeaks_median = np.median(waves["ECG_P_Peaks"])
            ppeaks_max = np.max(waves["ECG_P_Peaks"])
            ppeaks_min = np.min(waves["ECG_P_Peaks"])
            ppeaks_std = np.std(waves["ECG_P_Peaks"])
            ppeaks_mad = stats.median_abs_deviation(waves["ECG_P_Peaks"])

        except:
            #print(f"No result for {i}.")
            counter_p += 1
            ppeaks_mean = np.nan
            ppeaks_median = np.nan
            ppeaks_max = np.nan
            ppeaks_min = np.nan
            ppeaks_std = np.nan
            ppeaks_mad = np.nan

        try:
            # S-peak features
            speaks_mean = np.mean(waves["ECG_S_Peaks"])
            speaks_median = np.median(waves["ECG_S_Peaks"])
            speaks_max = np.max(waves["ECG_S_Peaks"])
            speaks_min = np.min(waves["ECG_S_Peaks"])
            speaks_std = np.std(waves["ECG_S_Peaks"])
            speaks_mad = stats.median_abs_deviation(waves["ECG_S_Peaks"])

        except:
            #print(f"No result for {i}.")
            counter += 1
            speaks_mean = np.nan
            speaks_median = np.nan
            speaks_max = np.nan
            speaks_min = np.nan
            speaks_std = np.nan
            speaks_mad = np.nan

        try:
            # T-peak features
            tpeaks_mean = np.mean(waves["ECG_T_Peaks"])
            tpeaks_median = np.median(waves["ECG_T_Peaks"])
            tpeaks_max = np.max(waves["ECG_T_Peaks"])
            tpeaks_min = np.min(waves["ECG_T_Peaks"])
            tpeaks_std = np.std(waves["ECG_T_Peaks"])
            tpeaks_mad = stats.median_abs_deviation(waves["ECG_T_Peaks"])

        except:
            #print(f"No result for {i}.")
            counter += 1
            tpeaks_mean = np.nan
            tpeaks_median = np.nan
            tpeaks_max = np.nan
            tpeaks_min = np.nan
            tpeaks_std = np.nan
            tpeaks_mad = np.nan

        #add features
        pqst_feat = [qpeaks_mean, qpeaks_median, qpeaks_max, qpeaks_min, qpeaks_std, qpeaks_mad,
            ppeaks_mean, ppeaks_median, ppeaks_max, ppeaks_min, ppeaks_std, ppeaks_mad,
            speaks_mean, speaks_median, speaks_max, speaks_min, speaks_std, speaks_mad,
            tpeaks_mean, tpeaks_median, tpeaks_max, tpeaks_min, tpeaks_std, tpeaks_mad]

        feature_PQST_peaks = np.vstack([feature_PQST_peaks, pqst_feat])

    #print(f"Counter is: {counter}.")

    feature_PQST_peaks = np.delete(feature_PQST_peaks, 0, 0)        #delete the (random) first column

    return feature_PQST_peaks

#Features:
    #Feature1: mean
    #Feature2: median
    #Feature3: max
    #Feature4: min
    #Feature5: STD
    #Feature6: MAD

def other_parameters(data_ts):

    features = np.empty(12)

    counter_skipped = 0 # for counting the skipped iterations

    for i in range(len(data_ts)):

        print(f"Iteration {i}")

        try:
            #capture the signal and clean it
            raw_signal = data_ts[i,][~np.isnan(data_ts[i,])]
            cleaned_signal = nk.ecg_clean(raw_signal, sampling_rate=300, method="neurokit")
            _, rpeaks = nk.ecg_peaks(cleaned_signal, sampling_rate=300)

            ecg_rate = nk.signal_rate(rpeak_values_cleaned, sampling_rate=300)

            ecg_rate_mean = np.mean(ecg_rate)
            ecg_rate_median = np.median(ecg_rate)
            ecg_rate_min = np.min(ecg_rate)
            ecg_rate_max = np.max(ecg_rate)
            ecg_rate_std = np.std(ecg_rate)
            ecg_rate_mad = stats.median_abs_deviation(ecg_rate)

        except:
            counter_skipped += 1
            ecg_rate_mean = np.nan
            ecg_rate_median = np.nan
            ecg_rate_min = np.nan
            ecg_rate_max = np.nan
            ecg_rate_std = np.nan
            ecg_rate_mad = np.nan

        try:
            rsp_rate = nk.ecg_rsp(ecg_rate, sampling_rate=300)

            rsp_rate_mean = np.mean(rsp_rate)
            rsp_rate_median = np.median(rsp_rate)
            rsp_rate_min = np.min(rsp_rate)
            rsp_rate_max = np.max(rsp_rate)
            rsp_rate_std = np.std(rsp_rate)
            rsp_rate_mad = stats.median_abs_deviation(rsp_rate)

        except:
            rsp_rate_mean = np.nan
            rsp_rate_median = np.nan
            rsp_rate_min = np.nan
            rsp_rate_max = np.nan
            rsp_rate_std = np.nan
            rsp_rate_mad = np.nan

        feat_i = [ecg_rate_mean, ecg_rate_median, ecg_rate_min, ecg_rate_max, ecg_rate_std, ecg_rate_mad,
            rsp_rate_mean, rsp_rate_median, rsp_rate_min, rsp_rate_max, rsp_rate_std, rsp_rate_mad]

        features = np.vstack([features, feat_i])

    print(f"Counter is: {counter_skipped}.")

    features = np.delete(features, 0, 0)

    return features

# HRV: heart rate variability
        #Feature 1: median_hr (bio-rpeaks)
        #Feature 2: median_hr_cleaned (nk-rpeaks)
        #Feature 3: hrv (=STD after zscore of bio-rpeaks)
        #Feature 4: hrv_cleaned (=STD after zscore of nk-rpeaks)
        #Feature 5: mean hr (bio-rpeaks)
        #Feature 6: mean hr_cleaned (nk-rpeaks)
        #Feature 7: min hr (bio-rpeaks)
        #Feature 8: min hr_cleaned (nk-rpeaks)
        #Feature 9: max hr (bio-rpeaks)
        #Feature10: max hr_cleaned (nk-rpeaks)
        #Feature11: median RR interval
        #Feature12: median RR_cleaned interval
        #Feature13: mean RR interval
        #Feature14: mean RR_cleaned interval
        #Feature15: min RR interval
        #Feature16: min RR_cleaned interval
        #Feature17: max RR interval
        #Feature18: max RR_cleaned interval
        #Feature19: STD RR interval
        #Feature20: STD RR_cleaned interval
        #Feature21: hrv_welch_HF
        #Feature22: hrv_welch_VHF
        #Feature23: hrv_welch_HFn
        #Feature24: hrv_welch_LnHF
        #Feature25: hrv_burg_HF
        #Feature26: hrv_burg_VHF
        #Feature27: hrv_burg_HFn
        #Feature28: hrv_burg_LnHF

"""
Information about features 21-28:
Frequency domain: Spectral power density in various frequency bands
(Ultra low/ULF, Very low/VLF, Low/LF, High/HF, Very high/VHF),
Ratio of LF to HF power, Normalized LF (LFn) and HF (HFn), Log transformed HF (LnHF).
"""

def feat_hrv(data_ts):

    feature_hr = []

    counter_skipped = 0

    for i in range(len(data_ts)):
        #capture the signal (clean and use neurokit.ecg_findpeaks & no clean and biosppy's ecg.engzee_segmenter)
        raw_signal = data_ts[i,][~np.isnan(data_ts[i,])]
        cleaned_signal = nk.ecg_clean(raw_signal, sampling_rate = 300, method='engzeemod2012')
        r_peaks = ecg.engzee_segmenter(raw_signal, 300)['rpeaks']
        r_peaks_cleaned = nk.ecg_findpeaks(cleaned_signal, 300)['ECG_R_Peaks']

        RR_intervals = np.diff(r_peaks) / 300                           #divide by the sampling frequency
        RR_intervals_cleaned = np.diff(r_peaks_cleaned) / 300           #divide by the sampling frequency

        heart_rate = 60/RR_intervals
        heart_rate_cleaned = 60/RR_intervals_cleaned

        peaks, info = nk.ecg_peaks(cleaned_signal, sampling_rate=300)
        hrv_welch = nk.hrv_frequency(peaks, sampling_rate=300, psd_method="welch")
        hrv_burg = nk.hrv_frequency(peaks, sampling_rate=300, psd_method="burg")

        #add HRs
        feature_hr.append([heart_rate, heart_rate_cleaned, RR_intervals, RR_intervals_cleaned,
            hrv_welch, hrv_burg])

    #extract properties/features
    feature_hrv = np.empty(28)

    for i in feature_hr:

        #delete noise at beginning/zscore of the hr
        zscore = stats.zscore(i[0], axis = 0, nan_policy = 'omit') #array with zscore for each element
        zscore_cleaned = stats.zscore(i[1], axis = 0, nan_policy = 'omit') #array with zscore for each element

        mean_hr = np.mean(i[0])
        mean_hr_cleaned = np.mean(i[1])

        # min & max
        try:
            min_hr = np.min(i[0])

        except ValueError:
            min_hr = np.nan

        try:
            min_hr_cleaned = np.min(i[1])

        except ValueError:
            min_hr_cleaned = np.nan

        try:
            max_hr = np.max(i[0])

        except ValueError:
            max_hr = np.nan

        try:
            max_hr_cleaned = np.max(i[1])

        except ValueError:
            max_hr_cleaned = np.nan

        z_threshold = 3

        for j in range(len(i[0])):
            if abs(zscore[j]) > z_threshold:
                i[0][j] = mean_hr

        for j in range(len(i[1])):
            if abs(zscore_cleaned[j]) > z_threshold:
                i[1][j] = mean_hr_cleaned

        #calculate the features
        hrv = np.std(i[0])
        hrv_cleaned = np.std(i[1])

        median_hr = np.median(i[0])
        median_hr_cleaned = np.median(i[1])

        RR_interval_median = np.median(i[2])
        RR_interval_mean = np.mean(i[2])

        try:
            RR_interval_min = np.min(i[2])

        except ValueError:
            RR_interval_min = np.nan

        try:
            RR_interval_max = np.max(i[2])

        except ValueError:
            RR_interval_max = np.nan

        RR_interval_std = np.std(i[2])

        RR_interval_cleaned_median = np.median(i[3])
        RR_interval_cleaned_mean = np.mean(i[3])

        try:
            RR_interval_cleaned_min = np.min(i[3])

        except ValueError:
            RR_interval_cleaned_min = np.nan

        try:
            RR_interval_cleaned_max = np.max(i[3])

        except ValueError:
            RR_interval_cleaned_max = np.nan

        RR_interval_cleaned_std = np.std(i[3])

        hrv_welch_HF = i[4]["HRV_HF"][0]
        hrv_welch_VHF = i[4]["HRV_VHF"][0]
        hrv_welch_HFn = i[4]["HRV_HFn"][0]
        hrv_welch_LnHF = i[4]["HRV_LnHF"][0]

        hrv_burg_HF = i[5]["HRV_HF"][0]
        hrv_burg_VHF = i[5]["HRV_VHF"][0]
        hrv_burg_HFn = i[5]["HRV_HFn"][0]
        hrv_burg_LnHF = i[5]["HRV_LnHF"][0]

        #add features
        hrv_feat = [median_hr, median_hr_cleaned, hrv, hrv_cleaned,
            mean_hr, mean_hr_cleaned, min_hr, min_hr_cleaned, max_hr, max_hr_cleaned,
            RR_interval_median, RR_interval_mean, RR_interval_min, RR_interval_max, RR_interval_std,
            RR_interval_cleaned_median, RR_interval_cleaned_mean, RR_interval_cleaned_min,
            RR_interval_cleaned_max, RR_interval_cleaned_std,
            hrv_welch_HF, hrv_welch_VHF, hrv_welch_HFn, hrv_welch_LnHF,
            hrv_burg_HF, hrv_burg_VHF, hrv_burg_HFn, hrv_burg_LnHF]

        feature_hrv = np.vstack([feature_hrv, hrv_feat])

    feature_hrv = np.delete(feature_hrv, 0, 0)

    #print(f"Counter skipped HRV: {counter_skipped}.")

    return feature_hrv

# Feature extraction pipeline

def feature_extraction(train_data_ts, test_data_ts):

    #calculate all the features
    #HRV
    feat_hrv_train = feat_hrv(train_data_ts)
    feat_hrv_test = feat_hrv(test_data_ts)
    print("finished with HRV-features")

    #R_peaks variance
    feat_rpeak_var_train = feat_rpeak_variance(train_data_ts)
    feat_rpeak_var_test = feat_rpeak_variance(test_data_ts)
    print("finished with R_peak_variance-features")

    #PQST_peaks
    feat_qst_peak_train = PQST_peaks(train_data_ts)
    feat_qst_peak_test = PQST_peaks(test_data_ts)
    print("finished with QST_peak-features")

    #other features
    #feat_other_train = other_parameters(train_data_ts)
    #feat_other_test = other_parameters(test_data_ts)
    #print("finished with other features")

    #Stack the features
    X_train_features = np.c_[feat_hrv_train, feat_rpeak_var_train, feat_qst_peak_train]
    X_test_features = np.c_[feat_hrv_test, feat_rpeak_var_test, feat_qst_peak_test]
    print("Feature extraction complete.")

    return pd.DataFrame.from_records(X_train_features), pd.DataFrame.from_records(X_test_features)

print(f"X_train: {X_train.shape} and X_test: {X_test.shape}.")

X_train_features, X_test_features = feature_extraction(X_train.to_numpy(), X_test.to_numpy())
