import pandas as pd

# Feature Extraction
import biosppy.signals.ecg as ecg
import neurokit2 as nk
import heartpy as hp

# Load datasets, decompressing pickle with gzip
X_train = pd.read_pickle('data/X_train.pkl', compression='gzip')
y_train = pd.read_pickle('data/y_train.pkl', compression='gzip')
X_test = pd.read_pickle('data/X_test.pkl', compression='gzip')

# Feature Selection

# R_peaks variance
    #Feature 1: number of spikes/total peaks
    #Feature 2: STD (after modified zscore)
    #Feature 3: MAD (after modified zscore)

# modified zscore based on MAD (median absolute deviation)
def modified_zscore(data, consistency_correction = 1.4826):
    median = np.median(data)
    deviation_from_med = np.array(data) - median

    mad = np.median(np.abs(deviation_from_med))
    mod_zscore = deviation_from_med/(consistency_correction*mad)

    return mod_zscore

def feat_rpeak_variance(data_ts, static_rpeak_threshold = 400):           #returns an array of the features

    feature_rpeaks_variance = np.empty(3)
    for i in range(len(data_ts)):

        #capture the signal (clean and use neurokit.ecg_findpeaks)
        raw_signal = data_ts[i,][~np.isnan(data_ts[i,])]
        cleaned_signal = nk.ecg_clean(raw_signal, sampling_rate = 300, method='engzeemod2012')
        r_peaks_cleaned = nk.ecg_findpeaks(cleaned_signal, 300)['ECG_R_Peaks']

        #extract the values of the peaks
        rpeak_values_cleaned = cleaned_signal[r_peaks_cleaned]

        #median & median
        median_rpeaks = np.median(rpeak_values_cleaned)
        mean_rpeak_cleaned = np.mean(rpeak_values_cleaned)

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

        #detect spikes (i.e. unusual high R_peaks)
        counter_spikes = 0
        for j in r_peaks_cleaned:
            if abs(cleaned_signal[j]) > abs((median_rpeaks + static_rpeak_threshold)):
                counter_spikes = counter_spikes + 1

        if len(r_peaks_cleaned) != 0:
            counter_spikes /= len(r_peaks_cleaned)                            #normalize the spikes with respect to the total number of R_peaks

        #add features
        var_feat = [counter_spikes, std_rpeaks, mad_rpeaks]
        feature_rpeaks_variance = np.vstack([feature_rpeaks_variance, var_feat])

    feature_rpeaks_variance = np.delete(feature_rpeaks_variance, 0, 0)        #delete the (random) first column

    print("R_peak variance feature calculated!")

    return feature_rpeaks_variance


# HRV: heart rate variability
        #Feature 1: median_hr (bio-rpeaks)
        #Feature 2: median_hr_cleaned (nk-rpeaks)
        #Feature 3: hrv (=STD after zscore of bio-rpeaks)
        #Feature 4: hrv_cleaned (=STD after zscore of nk-rpeaks)

def feat_hrv(data_ts):

    feature_hr = []

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

        #add HRs
        feature_hr.append([heart_rate, heart_rate_cleaned])


    #extract properties/features (mean)
    feature_hrv = np.empty(4)

    for i in feature_hr:

        #delete noise at beginning/zscore of the hr
        zscore = stats.zscore(i[0], axis = 0, nan_policy = 'omit') #array with zscore for each element
        zscore_cleaned = stats.zscore(i[1], axis = 0, nan_policy = 'omit') #array with zscore for each element

        mean_hr = np.mean(i[0])
        mean_hr_cleaned = np.mean(i[1])

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


        #add features
        hrv_feat = [median_hr, median_hr_cleaned, hrv, hrv_cleaned]
        feature_hrv = np.vstack([feature_hrv, hrv_feat])

    feature_hrv = np.delete(feature_hrv, 0, 0)

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

    #Stack the features
    X_train_features = np.c_[ feat_hrv_train, feat_rpeak_var_train ]
    X_test_features = np.c_[ feat_hrv_test, feat_rpeak_var_test ]

    return X_train_features, X_test_features
