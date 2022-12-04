import pandas as pd
import numpy as np
import scipy.stats as stats
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.impute import SimpleImputer

# Feature Extraction
import biosppy.signals.ecg as ecg
import neurokit2 as nk
import heartpy as hp

# Load datasets, decompressing pickle with gzip
X_train = pd.read_pickle('data/X_train.pkl', compression='gzip')
y_train = pd.read_pickle('data/y_train.pkl', compression='gzip')
X_test = pd.read_pickle('data/X_test.pkl', compression='gzip')

def remove_highly_correlated_features(X_train, X_test, threshold: float = 0.9, verbose: int = 1):
    """
    Remove highly correlated features from the dataset.
    :param X_train: training set
    :param X_test: test set
    :param threshold: threshold for the correlation between features above which the features are removed
    :param verbose: verbosity level
    :return: X_train, X_test without highly correlated features
    """
    if verbose >= 1:
        print(f"Removing highly correlated features")
    correlated_features = set()
    X_train_correlation_matrix = X_train.corr()
    for i in range(len(X_train_correlation_matrix.columns)):
        for j in range(i):
            if abs(X_train_correlation_matrix.iloc[i, j]) > threshold:
                colname = X_train_correlation_matrix.columns[i]
                correlated_features.add(colname)

    X_train = X_train.drop(labels=correlated_features, axis=1)
    X_test = X_test.drop(labels=correlated_features, axis=1)
    if verbose >= 2:
        print(f"{'':<1} Shape of the training set: {X_train.shape}")
    return X_train, X_test

def balance_classes(X_train, y_train):
    #smote = SMOTE()
    smote = BorderlineSMOTE(sampling_strategy = 'minority')
    # fit predictor and target variable
    x_smote, y_smote = smote.fit_resample(X_train, y_train)

    #r_under_sampler = RandomUnderSampler(random_state=42, replacement=True) # fit predictor and target variable
    #x_undersampled, y_undersampled = rus.fit_resample(X_train, y_train)

    return x_smote, y_smote

# Feature Selection

# R_peaks variance
    #Feature 1: number of spikes/total peaks
    #Feature 2: STD (after modified zscore)
    #Feature 3: MAD (after modified zscore)
    #Feature 4: peak median
    #Feature 5: peak mean
    #Feature 6: min peak
    #Feature 7: max peak

#P,Q,S,T-peaks and cardiac phase
#24 features in total
#For each peak type:
    #Feature1: mean
    #Feature2: median
    #Feature3: max
    #Feature4: min
    #Feature5: STD
    #Feature6: MAD

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

# modified zscore based on MAD (median absolute deviation)
def modified_zscore(data, consistency_correction = 1.4826):
    median = np.nanmedian(data)
    deviation_from_med = np.array(data) - median

    mad = np.nanmedian(np.abs(deviation_from_med))
    mod_zscore = deviation_from_med/(consistency_correction*mad)

    return mod_zscore

def all_features(data_ts, static_rpeak_threshold = 400):           #returns an array of the features

    feature_rpeaks_variance = np.empty(7)
    feature_PQST_peaks = np.empty(60)
    feature_hr = []
    feature_hrv = np.empty(28)

    for i in range(len(data_ts)):

        print(f"Iteration {i}")

        #capture the signal (clean and use neurokit.ecg_findpeaks)
        raw_signal = data_ts[i,][~np.isnan(data_ts[i,])]
        cleaned_signal = nk.ecg_clean(raw_signal, sampling_rate = 300, method='engzeemod2012')
        r_peaks_cleaned = nk.ecg_findpeaks(cleaned_signal, 300)['ECG_R_Peaks']

        #extract the values of the peaks
        rpeak_values_cleaned = cleaned_signal[r_peaks_cleaned]

        #median & mean
        median_rpeaks = np.nanmedian(rpeak_values_cleaned)
        mean_rpeak_cleaned = np.nanmean(rpeak_values_cleaned)

        #max
        try:
            max_rpeaks = np.nanmax(rpeak_values_cleaned)

        except ValueError:
            max_rpeaks = np.nan

        #min
        try:
            min_rpeaks = np.nanmin(rpeak_values_cleaned)

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
        std_rpeaks = np.nanstd(rpeak_values_cleaned)
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

        try:
            _, rpeaks = nk.ecg_peaks(cleaned_signal, sampling_rate=300)
            signals, waves = nk.ecg_delineate(cleaned_signal, rpeaks, sampling_rate=300, method="dwt")

            # Q-peak features
            qpeaks_mean = np.nanmean(waves["ECG_Q_Peaks"])
            qpeaks_median = np.nanmedian(waves["ECG_Q_Peaks"])
            qpeaks_max = np.nanmax(waves["ECG_Q_Peaks"])
            qpeaks_min = np.nanmin(waves["ECG_Q_Peaks"])
            qpeaks_std = np.nanstd(waves["ECG_Q_Peaks"])
            qpeaks_mad = stats.median_abs_deviation(waves["ECG_Q_Peaks"])

            # P-peak features
            ppeaks_mean = np.nanmean(waves["ECG_P_Peaks"])
            ppeaks_median = np.nanmedian(waves["ECG_P_Peaks"])
            ppeaks_max = np.nanmax(waves["ECG_P_Peaks"])
            ppeaks_min = np.nanmin(waves["ECG_P_Peaks"])
            ppeaks_std = np.nanstd(waves["ECG_P_Peaks"])
            ppeaks_mad = stats.median_abs_deviation(waves["ECG_P_Peaks"])
            ppeaks_onset_mean = np.nanmean(waves["ECG_P_Onsets"])
            ppeaks_offset_mean = np.nanmean(waves["ECG_P_Offsets"])
            ppeaks_onset_median = np.nanmedian(waves["ECG_P_Onsets"])
            ppeaks_offset_median = np.nanmedian(waves["ECG_P_Offsets"])

            ppeaks_onset_max = np.nanmax(waves["ECG_P_Onsets"])
            ppeaks_offset_max = np.nanmax(waves["ECG_P_Offsets"])
            ppeaks_onset_min = np.nanmin(waves["ECG_P_Onsets"])
            ppeaks_offset_min = np.nanmin(waves["ECG_P_Offsets"])
            ppeaks_onset_std = np.nanstd(waves["ECG_P_Onsets"])
            ppeaks_offset_std = np.nanstd(waves["ECG_P_Offsets"])
            ppeaks_onset_mad = stats.median_abs_deviation(waves["ECG_P_Onsets"])
            ppeaks_offset_mad = stats.median_abs_deviation(waves["ECG_P_Offsets"])

            # S-peak features
            speaks_mean = np.nanmean(waves["ECG_S_Peaks"])
            speaks_median = np.nanmedian(waves["ECG_S_Peaks"])
            speaks_max = np.nanmax(waves["ECG_S_Peaks"])
            speaks_min = np.nanmin(waves["ECG_S_Peaks"])
            speaks_std = np.nanstd(waves["ECG_S_Peaks"])
            speaks_mad = stats.median_abs_deviation(waves["ECG_S_Peaks"])

            # T-peak features
            tpeaks_mean = np.nanmean(waves["ECG_T_Peaks"])
            tpeaks_median = np.nanmedian(waves["ECG_T_Peaks"])
            tpeaks_max = np.nanmax(waves["ECG_T_Peaks"])
            tpeaks_min = np.nanmin(waves["ECG_T_Peaks"])
            tpeaks_std = np.nanstd(waves["ECG_T_Peaks"])
            tpeaks_mad = stats.median_abs_deviation(waves["ECG_T_Peaks"])
            tpeaks_onset_mean = np.nanmean(waves["ECG_T_Onsets"])
            tpeaks_offset_mean = np.nanmean(waves["ECG_T_Offsets"])
            tpeaks_onset_median = np.nanmedian(waves["ECG_T_Onsets"])
            tpeaks_offset_median = np.nanmedian(waves["ECG_T_Offsets"])
            tpeaks_onset_max = np.nanmax(waves["ECG_T_Onsets"])
            tpeaks_offset_max = np.nanmax(waves["ECG_T_Offsets"])
            tpeaks_onset_min = np.nanmin(waves["ECG_T_Onsets"])
            tpeaks_offset_min = np.nanmin(waves["ECG_T_Offsets"])
            tpeaks_onset_std = np.nanstd(waves["ECG_T_Onsets"])
            tpeaks_offset_std = np.nanstd(waves["ECG_T_Offsets"])
            tpeaks_onset_mad = stats.median_abs_deviation(waves["ECG_T_Onsets"])
            tpeaks_offset_mad = stats.median_abs_deviation(waves["ECG_T_Offsets"])

            # R-peaks onsets and offsets
            rpeaks_onset_mean = np.nanmean(waves["ECG_R_Onsets"])
            rpeaks_offset_mean = np.nanmean(waves["ECG_R_Offsets"])
            rpeaks_onset_median = np.nanmedian(waves["ECG_R_Onsets"])
            rpeaks_offset_median = np.nanmedian(waves["ECG_R_Onsets"])
            rpeaks_onset_max = np.nanmax(waves["ECG_R_Onsets"])
            rpeaks_offset_max = np.nanmax(waves["ECG_R_Onsets"])
            rpeaks_onset_min = np.nanmin(waves["ECG_R_Onsets"])
            rpeaks_offset_min = np.nanmin(waves["ECG_R_Onsets"])
            rpeaks_onset_std = np.nanstd(waves["ECG_R_Onsets"])
            rpeaks_offset_std = np.nanstd(waves["ECG_R_Onsets"])
            rpeaks_onset_mad = stats.median_abs_deviation(waves["ECG_R_Onsets"])
            rpeaks_offset_mad = stats.median_abs_deviation(waves["ECG_R_Onsets"])

        except:
            qpeaks_mean = np.nan
            qpeaks_median = np.nan
            qpeaks_max = np.nan
            qpeaks_min = np.nan
            qpeaks_std = np.nan
            qpeaks_mad = np.nan
            ppeaks_mean = np.nan
            ppeaks_median = np.nan
            ppeaks_max = np.nan
            ppeaks_min = np.nan
            ppeaks_std = np.nan
            ppeaks_mad = np.nan
            speaks_mean = np.nan
            speaks_median = np.nan
            speaks_max = np.nan
            speaks_min = np.nan
            speaks_std = np.nan
            speaks_mad = np.nan
            tpeaks_mean = np.nan
            tpeaks_median = np.nan
            tpeaks_max = np.nan
            tpeaks_min = np.nan
            tpeaks_std = np.nan
            tpeaks_mad = np.nan
            rpeaks_onset_mean = np.nan
            rpeaks_offset_mean = np.nan
            rpeaks_onset_median = np.nan
            rpeaks_offset_median = np.nan
            rpeaks_onset_max = np.nan
            rpeaks_offset_max = np.nan
            rpeaks_onset_min = np.nan
            rpeaks_offset_min = np.nan
            rpeaks_onset_std = np.nan
            rpeaks_offset_std = np.nan
            rpeaks_onset_mad = np.nan
            rpeaks_offset_mad = np.nan

        #add features
        pqst_feat = [qpeaks_mean, qpeaks_median, qpeaks_max, qpeaks_min, qpeaks_std, qpeaks_mad,
            ppeaks_mean, ppeaks_median, ppeaks_max, ppeaks_min, ppeaks_std, ppeaks_mad,
            speaks_mean, speaks_median, speaks_max, speaks_min, speaks_std, speaks_mad,
            tpeaks_mean, tpeaks_median, tpeaks_max, tpeaks_min, tpeaks_std, tpeaks_mad,
            ppeaks_onset_mean, ppeaks_offset_mean, ppeaks_onset_median, ppeaks_offset_median,
            ppeaks_onset_max, ppeaks_offset_max, ppeaks_onset_min, ppeaks_offset_min,
            ppeaks_onset_std, ppeaks_offset_std, ppeaks_onset_mad, ppeaks_offset_mad,
            tpeaks_onset_mean, tpeaks_offset_mean, tpeaks_onset_median, tpeaks_offset_median,
            tpeaks_onset_max, tpeaks_offset_max, tpeaks_onset_min, tpeaks_offset_min,
            tpeaks_onset_std, tpeaks_offset_std, tpeaks_onset_mad, tpeaks_offset_mad,
            rpeaks_onset_mean, rpeaks_offset_mean, rpeaks_onset_median, rpeaks_offset_median,
            rpeaks_onset_max, rpeaks_offset_max, rpeaks_onset_min, rpeaks_offset_min,
            rpeaks_onset_std, rpeaks_offset_std, rpeaks_onset_mad, rpeaks_offset_mad]

        feature_PQST_peaks = np.vstack([feature_PQST_peaks, pqst_feat])

        #print(f"Counter is: {counter}.")

        r_peaks = ecg.engzee_segmenter(raw_signal, 300)['rpeaks']

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

    for i in feature_hr:

        #delete noise at beginning/zscore of the hr
        zscore = stats.zscore(i[0], axis = 0, nan_policy = 'omit') #array with zscore for each element
        zscore_cleaned = stats.zscore(i[1], axis = 0, nan_policy = 'omit') #array with zscore for each element

        mean_hr = np.nanmean(i[0])
        mean_hr_cleaned = np.nanmean(i[1])

        # min & max
        try:
            min_hr = np.nanmin(i[0])

        except ValueError:
            min_hr = np.nan

        try:
            min_hr_cleaned = np.nanmin(i[1])

        except ValueError:
            min_hr_cleaned = np.nan

        try:
            max_hr = np.nanmax(i[0])

        except ValueError:
            max_hr = np.nan

        try:
            max_hr_cleaned = np.nanmax(i[1])

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
        hrv = np.nanstd(i[0])
        hrv_cleaned = np.nanstd(i[1])
        median_hr = np.nanmedian(i[0])
        median_hr_cleaned = np.nanmedian(i[1])
        RR_interval_median = np.nanmedian(i[2])
        RR_interval_mean = np.nanmean(i[2])

        try:
            RR_interval_min = np.nanmin(i[2])

        except ValueError:
            RR_interval_min = np.nan

        try:
            RR_interval_max = np.nanmax(i[2])

        except ValueError:
            RR_interval_max = np.nan

        RR_interval_std = np.nanstd(i[2])
        RR_interval_cleaned_median = np.nanmedian(i[3])
        RR_interval_cleaned_mean = np.nanmean(i[3])
        RR_interval_cleaned_std = np.nanstd(i[3])

        try:
            RR_interval_cleaned_min = np.nanmin(i[3])

        except ValueError:
            RR_interval_cleaned_min = np.nan

        try:
            RR_interval_cleaned_max = np.nanmax(i[3])

        except ValueError:
            RR_interval_cleaned_max = np.nan

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
    feature_PQST_peaks = np.delete(feature_PQST_peaks, 0, 0)                  #delete the (random) first column
    feature_rpeaks_variance = np.delete(feature_rpeaks_variance, 0, 0)        #delete the (random) first column

    return feature_rpeaks_variance, feature_PQST_peaks, feature_hrv

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

            ecg_rate_mean = np.nanmean(ecg_rate)
            ecg_rate_median = np.nanmedian(ecg_rate)
            ecg_rate_min = np.nanmin(ecg_rate)
            ecg_rate_max = np.nanmax(ecg_rate)
            ecg_rate_std = np.nanstd(ecg_rate)
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
            rsp_rate_mean = np.nanmean(rsp_rate)
            rsp_rate_median = np.nanmedian(rsp_rate)
            rsp_rate_min = np.nanmin(rsp_rate)
            rsp_rate_max = np.nanmax(rsp_rate)
            rsp_rate_std = np.nanstd(rsp_rate)
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


# Feature extraction pipeline

def feature_extraction(train_data_ts, test_data_ts):

    #calculate all the features
    train_rpeaks, train_pqst, train_hrv = all_features(train_data_ts)
    test_rpeaks, test_pqst, test_hrv = all_features(test_data_ts)

    #print(np.count_nonzero(~np.isnan(test_rpeaks)))
    #print(np.count_nonzero(~np.isnan(test_pqst)))
    #print(np.count_nonzero(~np.isnan(test_hrv)))

    #Stack the features
    X_train_features = np.c_[train_rpeaks, train_pqst, train_hrv]
    X_test_features = np.c_[test_rpeaks, test_pqst, test_hrv]
    print("Feature extraction complete.")

    return pd.DataFrame.from_records(X_train_features), pd.DataFrame.from_records(X_test_features) # convert back to pandas df

print(f"X_train: {X_train.shape} and X_test: {X_test.shape}.")


X_train = X_train.fillna(0)
X_smote, y_smote = balance_classes(X_train, y_train["y"])
print(X_smote)
print(y_smote)


#X_smote.to_pickle('X_train_smote.pkl', compression='gzip')
#y_smote.to_pickle('y_train_smote.pkl', compression='gzip')

"""
X_train_features, X_test_features = feature_extraction(X_train.to_numpy(), X_test.to_numpy())

X_train_features = X_train_features.dropna(axis = 1, how = 'all')
X_test_features = X_test_features.dropna(axis = 1, how = 'all')

X_train_features_index = X_train_features.copy().index
X_test_features_index = X_test_features.copy().index

X_train_features = pd.DataFrame(SimpleImputer().fit_transform(X_train_features), columns = X_train_features.columns, index = X_train_features_index)
X_test_features = pd.DataFrame(SimpleImputer().fit_transform(X_test_features), columns = X_test_features.columns, index = X_test_features_index)

X_train_removed, X_test_removed = remove_highly_correlated_features(X_train_features, X_test_features)

X_smote, y_smote = balance_classes(X_train_removed, y_train["y"])
"""
