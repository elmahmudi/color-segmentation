from scipy import signal
import numpy as np
import plotly.graph_objs as go


def band_stop(low, high, fs, sig):
    nyq = 0.5 * fs  # Nyquist Frequency
    low = low / nyq
    high = high / nyq
    b, a = signal.butter(N=6, Wn=[low, high], btype='bandstop')
    filtered_sig = signal.filtfilt(b, a, sig)
    return filtered_sig


def low_pass(fs, sig):
    fc = 100  # cutoff
    wn = fc / (fs / 2)
    # print(wn)
    b, a = signal.butter(N=3, Wn=wn, btype='low', analog=False)
    filtered_sig = signal.filtfilt(b, a, sig)
    return filtered_sig


def high_pass(fs, sig):
    fc = 1
    wn = fc / (fs / 2)
    # print(wn)
    b, a = signal.butter(N=3, Wn=wn, btype='highpass')
    filtered_sig = signal.filtfilt(b, a, sig)
    return filtered_sig


def filtering(signals, fs):
    filtered_signal = list()
    for signal_ in signals:
        # Get each filtered signal
        ###
        ch_ = high_pass(fs, signal_)
        ch_ = low_pass(fs, ch_)
        #fig = go.Figure()
        # fig.add_trace(go.Scatter(
        #     y=signal_,
        #     line=dict(shape='spline'),
        #     name='signal with noise'
        # ))
        # fig.add_trace(go.Scatter(
        #     y=ch_,
        #     line=dict(shape='spline'),
        #     name='filtered signal'
        # ))
        # fig.show()
        ch_ = band_stop(55, 65, fs, ch_)
        filtered_signal.append(ch_)
    return np.asarray(filtered_signal)
