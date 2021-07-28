import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dense, Activation, Flatten
import numpy as np
from keras import optimizers, utils
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from prepare_data import get_samples
from sklearn.preprocessing import LabelEncoder
from Validation import TestFunction
from collections import namedtuple
import h5py
import json

EEGBand = namedtuple("EEGBand", "band_name, min_freq, max_freq")
# Settings of general parameter
metadata = {'eeg_bands': [['Delta', 0, 4], ['Theta', 4, 8], ['Alpha', 8, 16], ['Beta', 16, 32], ['Gamma', 32, 80]],
            'eeg_channels': ['Fp2', 'F8', 'T8', 'P8', 'O2', 'F3', 'Fz', 'F4', 'C3', 'C5', 'C4', 'P3', 'Pz', 'P4', 'Fp1',
                             'F7', 'T7', 'P7', 'O1'], 'scoring_period': 8.0, 'scoring_period_overlap': 0.0, 'ignore': 2}


class model_builder:

    def read_signal(self, path):
        X_frames, Y_train = get_samples(path, metadata)
        reshape = (-1, len(metadata['eeg_channels']), len(metadata['eeg_bands']))
        train_X = np.array(X_frames).reshape(reshape)
        return train_X, Y_train

    def training_stage(self, data, labels, Epoch, Batch_Size, name):
        x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, shuffle=True)
        model = Sequential()

        model.add(Conv1D(64, 3, padding='same', input_shape=x_train.shape[1:]))
        model.add(Activation('relu'))

        model.add(Conv1D(64, 2, padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_size=2, padding='same'))

        model.add(Conv1D(64, 2, padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_size=2, padding='same'))

        model.add(Flatten())

        model.add(Dense(16))
        model.add(Activation('relu'))

        model.add(Dense(2))
        model.add(Activation('softmax'))

        model.summary()

        opt = optimizers.Adam(learning_rate=0.0001)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

        start = time.time()
        history = model.fit(x_train, y_train, batch_size=Batch_Size, validation_data=(x_test, y_test), epochs=Epoch)

        duration = time.time() - start
        print('Elapsed time', duration)

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epoch1 = history.epoch

        # Save the model as h5 with its accuracy
        model_name = str("{:.2f}".format(val_acc[len(val_acc)-1])) + '_' + name
        model.save(model_name)

        #
        hf = h5py.File(model_name, mode='a')
        hf.attrs['ni2o_meta'] = json.dumps(metadata)

        # Plot the accuracy and loss curves
        plt.plot(epoch1, acc, 'b', label='Training acc')
        plt.plot(epoch1, val_acc, 'r', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()

        plt.figure()

        plt.plot(epoch1, loss, 'b', label='Training loss')
        plt.plot(epoch1, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()

        plt.show()

        return model_name


if __name__ == '__main__':

    # Choose a name for the model i.e Selection_1 or _2 ...
    model_name = 'Depression_19.h5'

    # Enter your training data path
    files_path = 'UTData\\train\\'
    X, Y = model_builder().read_signal(files_path)
    #Save processed  data into file in order to use it for retrain the model later
    np.savez_compressed('DepressionUTData.npz', X, Y)

    # Load data if saved it
    # data = np.load('DepressionUTData.npz')
    # X, Y = data['arr_0'], data['arr_1']

    # Code labels to be binary
    out_encoder = LabelEncoder()
    out_encoder.fit(Y)
    train_Y = out_encoder.transform(Y)
    train_Y = utils.to_categorical(train_Y, num_classes=2)

    # Setup training parameters
    epoch = 150
    Batch_size = 256
    model_name = model_builder().training_stage(X, train_Y, epoch, Batch_size, model_name)
    print('Training is done and the model is saved')

    # ####### Test part
    # Test path
    test_path = 'UTData\\test\\MDD\\'
    TestFunction(test_path, model_name)
