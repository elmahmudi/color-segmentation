from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import FastICA
from pywt import dwt, threshold, idwt

# pip freeze > requirements.txt
# pip install -r requirements.txt
# pip install pywavelets


def apply_dwt(sig):
    # Discrete Wavelet Transform (DWT) to wavelet transform the ICs

    (cA, cD) = dwt(data=sig, wavelet='db2')
    scalar = 0.25
    sigma = np.median(np.abs(cA)) / 0.6745
    thresh = np.sqrt(2 * np.log(len(cA))) * sigma * scalar
    cA = threshold(cA, thresh, mode='soft')

    sigma = np.median(np.abs(cD)) / 0.6745
    thresh = np.sqrt(2 * np.log(len(cD))) * sigma * scalar
    cD = threshold(cD, thresh, mode='soft')

    # Reconstruct cleaned components
    cleaned_data = idwt(cA, cD, wavelet='db2')

    return cleaned_data


def plotting(raw, high, low, stop, cleaned_ica, cleaned_wica, fs, cha, length):
    # Plotting original and cleaned signals
    w, l1 = cleaned_wica[:, :length].shape
    time_step = 1 / fs
    time_vec = np.arange(0, l1 / fs, time_step)

    # Plotting original signal
    i = 0
    plt.figure('Raw signal')
    for ch in raw[:, :l1]:
        ch = ch.T
        plt.subplot(w, 1, i + 1)
        plt.plot(time_vec, ch)
        plt.ylabel(cha[i])
        i += 1

    # Plotting filtered signal by high pass
    i = 0
    plt.figure('filtered signal by high pass')
    for ch in high[:, 0:l1]:
        ch = ch.T
        plt.subplot(w, 1, i + 1)
        plt.plot(time_vec, ch)
        plt.ylabel(cha[i])
        i += 1

    # Plotting cleaned signal by ICA
    i = 0
    plt.figure('filtered signal by low pass')
    for ch in low[:, 0:l1]:
        ch = ch.T
        plt.subplot(w, 1, i + 1)
        plt.plot(time_vec, ch)
        plt.ylabel(cha[i])
        i += 1
    # Plotting cleaned signal by ICA
    i = 0
    plt.figure('filtered signal by band stop')
    for ch in stop[:, 0:l1]:
        ch = ch.T
        plt.subplot(w, 1, i + 1)
        plt.plot(time_vec, ch)
        plt.ylabel(cha[i])
        i += 1
    # Plotting cleaned signal by ICA
    i = 0
    plt.figure('cleaned signal by ICA')
    for ch in cleaned_ica[:, 0:l1]:
        ch = ch.T
        plt.subplot(w, 1, i + 1)
        plt.plot(time_vec, ch)
        plt.ylabel(cha[i])
        i += 1

    i = 0
    plt.figure('cleaned signal by WICA')
    for ch in cleaned_wica[:, 0:l1]:
        ch = ch.T
        plt.subplot(w, 1, i + 1)
        plt.plot(time_vec, ch)
        plt.ylabel(cha[i])
        i += 1
    plt.show()


# Fast ICA
def fast_ica(eeg_):
    eeg_ = eeg_.T
    _, n_com = np.shape(eeg_)
    # Applying FAST ICA for EEG raw data
    ica = FastICA(n_components=n_com, max_iter=1000, tol=0.04)
    ica.fit(eeg_)
    # Get ICA components
    components = ica.transform(eeg_)

    # Compute a sparsity value for each ICA component
    sp = np.zeros(n_com)
    for s in range(n_com):
        y = components[:, s]
        sp[s] = (np.max(np.abs(y)) / np.std(y)) * np.log(np.std(y) / np.median(np.abs(y)))

    # Remove unwanted components
    components[:, np.where((sp >= 8.7))] = 0

    # rebuild signal
    ica_cleaned_signal = ica.inverse_transform(components)

    return ica_cleaned_signal.T


# WICA
def wICA(eeg_):
    eeg_ = eeg_.T

    ica = FastICA(max_iter=1000, tol=0.04)
    ica.fit(eeg_)
    components = ica.transform(eeg_)
    IC = components.T

    wIc = list()
    for s in range(len(IC)):
        sig = IC[s, :]
        # Apply Discrete Wavelet Transform (DWT) to wavelet transform the ICs
        Y = apply_dwt(sig)
        # Y = apply_swt(sig)
        wIc.append(Y)

    wIc = np.asarray(wIc)  # Reconstruct a wavelet IC (wIC)

    # nn = IC - wIc
    wica_cleaned_signal = ica.inverse_transform(wIc.T)

    return wica_cleaned_signal.T