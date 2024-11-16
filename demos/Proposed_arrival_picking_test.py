# Import python library
from keras.models import load_model
from keras import backend as K
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from scipy.signal import butter, lfilter, lfilter_zi
import tensorflow as tf
from utils import *

seed = 202410
np.random.seed(seed)
tf.random.set_seed(seed)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)

import os

# Make a new dir to save the picking error
directory_path = "test_results_proposed"

# Create the directory, including any necessary parent directories
os.makedirs(directory_path, exist_ok=True)

## Load data and calculate the arrivals
# Load the TXED dataset
# please down load the TXED from: https://drive.google.com/drive/folders/1WXVB8ytNB4bOaZ97oq6OmMRyAEg95trp?usp=sharing
f = h5py.File("/home/g202321530/Yang/Data/Earthquake_data/TXED/TXED_0913.h5", 'r')
# Load randomly selected random id
event_id = np.load('./data/signalid_random_1.5w.npy')


# obtain the P- and S-wave arrivals
P_arrival_list = []
S_arrival_list = []
print('-----------arrival time calculation begin-------------------')
for key in event_id:
    if key in f:   
        dataset = f.get(key)
        P_arrival_list.append(int(dataset.attrs['p_arrival_sample']))
        S_arrival_list.append(int(dataset.attrs['s_arrival_sample']))
P_arrival_list = np.array(P_arrival_list)
S_arrival_list = np.array(S_arrival_list)
P_phase_label = P_arrival_list
S_phase_label = S_arrival_list

print(P_arrival_list.shape, P_arrival_list)
print(S_arrival_list.shape, S_arrival_list)
print('-----------arrival time calculation end-------------------')

# band-pass and normalization of the 3-C waveforms
signal_list = []
print('-----------signal format convert begin-------------------')
for key in event_id:
    if key in f:   
        dataset = f.get(key)
        datas = dataset['data']
        datas = np.array(datas)
        datas_0 = butter_bandpass_filter_zi(datas[:,0], 1, 45, 100, order=3)
        datas_1 = butter_bandpass_filter_zi(datas[:,1], 1, 45, 100, order=3)
        datas_2 = butter_bandpass_filter_zi(datas[:,2], 1, 45, 100, order=3)
        datas = np.vstack([datas_0, datas_1, datas_2])
        signal_list.append(datas) 
signal_values = np.array(signal_list)
bp_signal= np.transpose(signal_values, [0, 2, 1])


#Normalized trace-by-trace
max_values_per_event = np.max(bp_signal, axis=1)
# Normalize each component of each event by dividing by its maximum value
normalized_phase_data = bp_signal / max_values_per_event[:, np.newaxis, :]
print('-----------signal format convert finish-------------------')
print(bp_signal.shape)

## Load the pre-trained model and perform the inference
learning_rate = 0.001  # Specify your learning rate

P_phase_model = load_model('./model/P_wave_phase_picking_10w_random_1006_256_100.h5')
S_phase_model = load_model('./model/S_wave_phase_picking_10w_random_1006_256_200.h5')

P_phase_output = P_phase_model.predict(normalized_phase_data)
S_phase_output = S_phase_model.predict(normalized_phase_data)

## Calculating and saving the predicted error of the proposed model
Proposed_P_error_indx = evaluate_picking(P_phase_output, P_phase_label)
Proposed_S_error_indx = evaluate_picking(S_phase_output, S_phase_label)

## Saving the results
np.save('./test_results_proposed/Pwave_error_proposed.npy', Proposed_P_error_indx)
np.save('./test_results_proposed/Swave_error_proposed.npy', Proposed_S_error_indx)