import numpy as np
from numpy import arange, abs
from os import listdir
from scipy.fft import rfft, rfftfreq
# from filters import filtering
import mne


def get_bands(segment, eeg_bands, fs):
    data = abs(rfft(segment)) / len(segment)

    # Compute the frequencies
    freq = rfftfreq(len(segment), 1.0 / fs)

    # Compute the bands for each amplitude
    resulted_band = dict()
    for band in eeg_bands:
        get_idx = np.where((freq >= band[1]) & (freq <= band[2]))
        resulted_band[band[0]] = np.mean(data[get_idx])
    return resulted_band


def get_steps(samples, metadata, fs):
    # Divide the signal into overlapping steps
    i = 0
    period = []
    chunk_per_frame = fs * metadata['scoring_period']
    while i + chunk_per_frame <= samples:
        period.append((i, i + chunk_per_frame))
        i = i + chunk_per_frame - int(chunk_per_frame * metadata['scoring_period_overlap'])
    return period


def get_frames(signals, metadata, fs):
    # Create frames for any number of bands
    frames = []
    _, sig = np.shape(signals)
    periods = get_steps(sig, metadata, fs)
    for i in range(len(periods)):
        chs = []
        for channel in signals:
            segment = np.array(channel[int(periods[i][0]):int(periods[i][1])])
            bands = get_bands(segment, metadata['eeg_bands'], fs)
            k = 0
            frame = np.zeros(len(metadata['eeg_bands']))
            for band in bands:
                frame[k] = bands[band]
                k += 1
            chs.append(frame.tolist())

        frames.append(chs)
    return np.array(frames)


def get_samples(path, metadata):
    # Build a set of frames for each subject
    sel_channels = metadata['eeg_channels']
    frames = []
    labels = []
    for SubFolder in listdir(path):
        file_names = path + SubFolder + '\\'
        print('Dealing with class No: ', SubFolder)
        i = 0
        for file in listdir(file_names):
            i = i + 1

            print('Processing: ', file, '. (', i, ' of ', len(listdir(file_names)), ')')
            #########
            file = file_names + file
            raw = mne.io.read_raw_bdf(file)
            raw = raw.load_data()
            org_channels = raw.ch_names
            signals = raw['data'][0]
            fs = raw.info['sfreq']
            # #########
            if fs != 250:
                # Resampling
                fs = 250
                downsampled_raw = raw.copy().resample(sfreq=fs)
                signals = downsampled_raw['data'][0]
            # #########

            select_cha = list()
            for j in arange(len(sel_channels)):
                select_cha.append(signals[org_channels.index(sel_channels[j])])

            select_cha = np.asarray(select_cha)
            select_cha = select_cha[:, metadata['ignore'] * fs:]
            # select_cha= filtering(select_cha, fs)
            frame = get_frames(select_cha, metadata, fs)
            for label in range(len(frame)): labels.append(SubFolder)
            frames.extend(np.array(frame))

    return frames, labels
