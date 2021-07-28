import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from filters import filtering
from os import listdir
from keras.models import load_model
from WICA_cleaner import *
import mne
from numpy import arange
import h5py
import json
from scipy.fft import rfft, rfftfreq


def partition(ranges, n):
    n = int(n)
    # Generate n ranges of chunks based on a length of a signal
    for i in range(0, len(ranges), n):
        yield ranges[i:i + n]
    return ranges


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


def get_frame(signal, Fs, metadata):

    frames = []
    for channel in range(len(signal)):
        segment = signal[channel, :]
        bands = get_bands(segment, metadata['eeg_bands'], Fs)
        k = 0
        frame = np.zeros(len(metadata['eeg_bands']))
        for band in bands:
            frame[k] = bands[band]
            k += 1
        frames.append(frame.tolist())

    return np.array(frames)


def read_file(file, metadata):
    # Read file data
    sel_channels = metadata['eeg_channels']
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

    # Make a length of the signal even for wavelet
    _, sig = np.shape(select_cha)
    if sig % 2 != 0:
        select_cha = select_cha[:, 0: sig - 1]

    return select_cha, fs


def diagnosis(signals, model, metadata, Fs):
    ch, sig = np.shape(signals)
    # Create a set of segments
    segments = list(partition(range(0, sig), metadata['scoring_period'] * int(Fs)))

    median_classification = np.zeros(len(segments))
    bands_size = len(metadata['eeg_bands'])
    k = 0
    for i in range(len(segments)):
        chunk = get_frame(signals[:, segments[i].start:segments[i].stop], Fs, metadata)
        #one_chunk = bands(signals[:, segments[i].start:segments[i].stop], Fs, dictionary)
        # Make a prediction for each chunk
        output = model.predict(np.array(chunk).reshape(-1, ch, len(metadata['eeg_bands'])))
        median_classification[k] = output[:, 1] * 100
        k = k + 1

    return np.average(median_classification)


def TestFunction(path, model_name):
    spreadsheetname = model_name[0:len(model_name) - 2] + '.csv'
    with h5py.File(model_name, 'r') as f:
        metadata = json.loads(f.attrs['ni2o_meta'])
    i = 0
    new_list = (['ID', 'Result'])

    # Load the model
    model = load_model(model_name)

    for file_ in listdir(path):
        lists = list()
        i = i + 1
        print('\nProcessing: ', file_, '. (', i, ' of ', len(listdir(path)), ')')
        #########
        raw, fs = read_file(path + file_, metadata)
        # Apply what you want here (filters ICA, or WICA)
        # raw = filtering(raw, fs)
        # raw = fast_ica(raw)
        # raw = wICA(raw)

        res = diagnosis(raw, model, metadata, fs)
        file_name = file_[0:len(file_) - 4]
        lists.append(file_name)
        lists.append(float("{:.2f}".format(res)))
        #print('Percentage = ', "{:.2f}".format(res))
        new_list = np.vstack((new_list, lists))
    np.savetxt(spreadsheetname, new_list, delimiter=", ", fmt='% s')
    print(' The results are saved in the Results.csv in sheet name: ', model_name)
