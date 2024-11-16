import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from keras.callbacks import LearningRateScheduler
from tensorflow.keras import regularizers 
from tensorflow.keras.losses import binary_crossentropy 
from keras import backend as K
import tensorflow as tf
from scipy import stats
from scipy.signal import butter, lfilter, lfilter_zi



## Define a band-pass filter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter_zi(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    zi = lfilter_zi(b, a)
    y,zo = lfilter(b, a, data, zi=zi*data[0])
    return y


def generate_first_arrival_labels(p_indx, s_indx, sample_num):
    """
    Generate labels for P-wave and S-wave first arrival picking.

    Parameters:
        p_indx (float): Arrival time of P-wave.
        s_indx (float): Arrival time of S-wave.
        sample_num (float): Total duration of the seismic signal.

    Returns:
        np.array: Array of labels indicating P-wave, S-wave, or neither.
    """
    all_time = np.linspace(0, sample_num, sample_num)
    #print(sample_num, all_time.shape)
    p_wave_arrival_time = p_indx  # P-wave arrival time (in seconds)
    s_wave_arrival_time = s_indx  # S-wave arrival time (in seconds)
    
    # Create Gaussian-shaped labels for P-wave and S-wave arrivals
    sigma = 10 # Standard deviation of the Gaussian
    p_wave_label = np.exp(-(all_time - p_wave_arrival_time)**2 / (2 * sigma**2))  # Gaussian label for P-wave
    s_wave_label = np.exp(-(all_time - s_wave_arrival_time)**2 / (2 * sigma**2))  # Gaussian label for S-wave

    return p_wave_label, s_wave_label


## Deine the evaluation metrics
def evaluate_picking(pred_array, label_array):
    pred_indx_array = []
    label_indx_array = []
    for i in range (pred_array.shape[0]-1):
        pred_indx = np.argmax(pred_array[i, :])
        # label_indx = np.argmax(label_array[i, :])
        label_indx = label_array[i]
        pred_indx_array.append(pred_indx)
        label_indx_array.append(label_indx)

    pred_indx_array = np.array(pred_indx_array)
    label_indx_array = np.array(label_indx_array)

    # calculate the error index
    err_indx_array = label_indx_array- pred_indx_array

    return err_indx_array

def cal_mae_std(input_array, m):
    # from sample to seconds
    input_array = input_array/100
    # filter those outliers
    input_array[(input_array > m) | (input_array < -m)] = 0

    # calculate the evaluation metrics
    mae = np.mean(np.abs(input_array - np.mean(input_array)))
    std = np.std(input_array)

    # calculate the precision of picking results (within ±1 s)
    precision = (np.sum(np.abs(input_array) <= 1) / len(input_array)) * 100
    
    return input_array, mae, std, precision 

def denoising_loss(y_true, y_pred):    
    mse_loss = K.mean(K.square(y_true - y_pred), axis=-1)  
    ssim_loss = 1 - K.mean(K.ssim(y_true, y_pred, axis=-1), axis=-1)  
    alpha = 0.5 
    loss = alpha * mse_loss + (1 - alpha) * ssim_loss  
  
    return loss

''' Display slices '''
def plot_metrics(history):
    ''' Plot metrics history during training '''    
    # list all data in history
    #print(history.keys())
    
    # summarize history for accuracy
    plt.subplots(1, 2, figsize=(14,6))
    plt.subplot(121)
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('Model accuracy',fontsize=20)
    plt.ylabel('Accuracy',fontsize=20)
    plt.xlabel('Epoch',fontsize=20)
    plt.legend(['train', 'test'], loc='center right',fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.tick_params(axis='both', which='minor', labelsize=18)
    
    # summarize history for loss
    plt.subplot(122)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model loss',fontsize=20)
    plt.ylabel('Loss',fontsize=20)
    plt.xlabel('Epoch',fontsize=20)
    plt.legend(['train', 'test'], loc='center right',fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.tick_params(axis='both', which='minor', labelsize=18)
    plt.show()


def plot_acc_epoch():
    ''' Plot accuracy and loss for training and validation data '''
    # Accuracy
    train_acc = history.history['accuracy']  
    train_loss = history.history['loss']  
  
    val_acc = history.history['val_accuracy']  
    val_loss = history.history['val_loss']  
    
    epochs = range(1, len(train_acc) + 1)  
    plt.subplots(1, 2, figsize=(12, 5))
    plt.subplot(121)
    plt.plot(epochs,train_acc,'b-',lw=1.5,label='Training acc')
    plt.plot(epochs,val_acc,  'r-',lw=1.5,label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    # Loss
    plt.subplot(122)
    plt.plot(train_loss,'b-',lw=1.5,label='Training loss')
    plt.plot(val_loss,  'r-',lw=1.5,label='Validation loss')    
    plt.title('Training and validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()

# define threshold condition
def class_thres(arr, alpha):  
    for i in range(arr.shape[0]):  
        for j in range(arr.shape[1]):  
            arr[i, j] = 1 if arr[i, j] > alpha else 0  
      
    return arr 


#define learning rate decay strategy
def lr_schedule(epoch):
    initial_lr = 1e-3

    if epoch <= 20:
        lr = initial_lr
    elif epoch <= 40:
        lr = initial_lr / 2
    elif epoch <= 60:
        lr = 3e-4
    elif epoch <= 80:
        lr = initial_lr / 10
    else:
        lr = initial_lr / 20
   # print('Learning rate: ', lr)
    return lr
lr_scheduler = LearningRateScheduler(lr_schedule)



def lr_schedule1(epoch):  
    initial_learning_rate = 0.001  
    if epoch <= 20:  
        lr = initial_learning_rate  
    elif 20 < epoch <= 50:  
        lr = initial_learning_rate / 5 
    else:  
        lr = (initial_learning_rate / 5) / (2 ** ((epoch - 51) // 10))  
    return lr  
lr_scheduler1 = LearningRateScheduler(lr_schedule1)

def lr_schedule2(epoch):  
    initial_learning_rate = 0.0001  
    if epoch <= 20:  
        lr = initial_learning_rate  
    elif 20 < epoch <= 50:  
        lr = initial_learning_rate / 5 
    else:  
        lr = (initial_learning_rate / 5) / (2 ** ((epoch - 51) // 10))  
    return lr  
lr_scheduler2 = LearningRateScheduler(lr_schedule2)


def lr_schedule3(epoch):  
    initial_learning_rate = 0.001  
    if epoch <= 20:  
        lr = initial_learning_rate  
    elif 20 < epoch <= 50:  
        lr = initial_learning_rate / 5 
    else:  
        # Calculate the number of intervals of 50 epochs beyond epoch 50
        num_intervals = (epoch - 50) // 50 + 1
        # Calculate the learning rate decay factor
        decay_factor = 2 ** num_intervals
        # Calculate the learning rate
        lr = (initial_learning_rate / 5) / decay_factor
    return lr  

lr_scheduler3 = LearningRateScheduler(lr_schedule3)

        
## Define a band-pass filter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter_zi(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    zi = lfilter_zi(b, a)
    y,zo = lfilter(b, a, data, zi=zi*data[0])
    return y


## Deine the evaluation metrics
def evaluate_picking(pred_array, label_array):
    pred_indx_array = []
    label_indx_array = []
    for i in range (pred_array.shape[0]-1):
        pred_indx = np.argmax(pred_array[i, :])
        # label_indx = np.argmax(label_array[i, :])
        label_indx = label_array[i]
        pred_indx_array.append(pred_indx)
        label_indx_array.append(label_indx)

    pred_indx_array = np.array(pred_indx_array)
    label_indx_array = np.array(label_indx_array)

    # calculate the error-index
    err_indx_array = label_indx_array- pred_indx_array

    return err_indx_array

def cal_mae_std(input_array, m):
    # from sample to seconds
    input_array = input_array/100
    # filter those outliers
    input_array[(input_array > m) | (input_array < -m)] = 0

    # calculate the evaluation metrics
    mae = np.mean(np.abs(input_array - np.mean(input_array)))
    std = np.std(input_array)

    # calculate the precision of picking results (within ±1 s)
    precision = (np.sum(np.abs(input_array) <= 1) / len(input_array)) * 100
    
    return input_array, mae, std, precision 

import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

nmi = normalized_mutual_info_score
ari = adjusted_rand_score


def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    print(y_pred.size, y_true.size, y_pred.shape, y_true.shape)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
     
    from scipy.optimize import linear_sum_assignment as linear_assignment 
 
 
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def mean(x):
    return np.mean(x, axis=a)

def std(x):
    return np.std(x, axis=a)

def ptp(x):
    return np.ptp(x, axis=a)

def var(x):
    return np.var(x, axis=a)

def minim(x):
    return np.min(x, axis=a)

def maxim(x):
    return np.max(x, axis=a)

def argminim(x):
    return np.argmin(x, axis=a)

def argmaxim(x):
    return np.argmax(x, axis=a)

def rms(x):
    return np.sqrt(np.mean(x**2, axis=a))

def abs_diff_signal(x):
    return np.sum(np.abs(np.diff(x, axis=a)), axis=a)
    
def skewness(x):
    return stats.skew(x, axis=a)

def kurtosis(x):
    return stats.kurtosis(x, axis=a)

def concatenate_features(x):
    return np.concatenate((mean(x), std(x), ptp(x), var(x), minim(x), maxim(x), argminim(x), argmaxim(x),
                         rms(x), abs_diff_signal(x), skewness(x), kurtosis(x)), axis=a)
                         
## define signal feature 2                         
def mean1(x):
    return np.mean(x)

def std1(x):
    return np.std(x)

def ptp1(x):
    return np.ptp(x)

def var1(x):
    return np.var(x)

def minim1(x):
    return np.min(x)

def maxim1(x):
    return np.max(x)

def argminim1(x):
    return np.argmin(x)

def argmaxim1(x):
    return np.argmax(x)

def rms1(x):
    return np.sqrt(np.mean(x**2))

def abs_diff_signal1(x):
    return np.sum(np.abs(np.diff(x)))
    
def skewness1(x):
    return stats.skew(x)

def kurtosis1(x):
    return stats.kurtosis(x)

def concatenate_features1(x):
    return print("mean:",'%.3f'% mean1(x), 'std:','%.3f'% std1(x), 'ptp:', '%.3f'% ptp1(x), 'var:', '%.3f'% var1(x), 'minim:', '%.3f'% minim1(x), 'maxim:', '%.3f'% maxim1(x), 'argminim:', '%.3f'% argminim1(x), 'argmaxim:', '%.3f'% argmaxim1(x), 'rms:', '%.3f'% rms1(x), 'abs_diff_signal1:', '%.3f'% abs_diff_signal1(x), 'skewness1:', '%.3f'% skewness1(x), 'kurtosis1:', '%.3f'% kurtosis1(x))

def pwave_signal_label_visualization(signal, p_predict, p_labels, sample_num, err_indx):
    data = signal
    p_predict_indx = np.argmax(p_predict)
    p_labels_indx = np.argmax(p_labels)
    time = np.linspace(0, sample_num, sample_num)
    fig = plt.figure(figsize=(6, 6))
    ax1 = fig.add_subplot(511)
    plt.plot(time, data[:,0], 'k',label='Z')
    ymin,yma = ax1.get_ylim()
    # plt.vlines(spt,ymin,yma,color='r',linewidth=2)
    # plt.vlines(sst,ymin,yma,color='b',linewidth=2)
    legend_properties = {'weight':'bold'}
    ymin, ymax = ax1.get_ylim()
    plt.legend(loc = 'upper right', borderaxespad=0., prop=legend_properties)
    plt.vlines(p_labels_indx, ymin+0.2, ymax-0.2, color='deepskyblue', linewidth=2, label='P-arrival')
    plt.vlines(p_predict_indx, ymin, ymax, color='tomato', linewidth=2, linestyle='--', label='P-arrival label')
    plt.legend(loc = 'upper right', borderaxespad=0., prop=legend_properties)
    plt.ylabel('Amplitude', fontsize=12)
    plt.xlim([0, 6000])
    ax1.set_xticklabels([])
    
    ax = fig.add_subplot(512) 
    plt.plot(time, data[:,1], 'k',label='N')
    ymin,yma = ax1.get_ylim()
    # plt.vlines(spt,ymin,yma,color='r',linewidth=2)
    # plt.vlines(sst,ymin,yma,color='b',linewidth=2)
    legend_properties = {'weight':'bold'}
    ymin, ymax = ax.get_ylim()
    plt.legend(loc = 'upper right', borderaxespad=0., prop=legend_properties)
    plt.vlines(p_labels_indx, ymin+0.2, ymax-0.2, color='deepskyblue', linewidth=2, label='P-arrival')
    plt.vlines(p_predict_indx, ymin, ymax, color='tomato', linewidth=2, linestyle='--')
    plt.ylabel('Amplitude', fontsize=12) 
    plt.xlim([0, 6000])
    ax.set_xticklabels([])
    
    ax = fig.add_subplot(513) 
    plt.plot(time, data[:,2], 'k',label='E')
    ymin,yma = ax1.get_ylim()
    # plt.vlines(spt,ymin,yma,color='r',linewidth=2)
    # plt.vlines(sst,ymin,yma,color='b',linewidth=2)
    legend_properties = {'weight':'bold'}
    ymin, ymax = ax.get_ylim()
    plt.legend(loc = 'upper right', borderaxespad=0., prop=legend_properties)
    plt.vlines(p_labels_indx, ymin+0.2, ymax-0.2, color='deepskyblue', linewidth=2, label='P-arrival')
    plt.vlines(p_predict_indx, ymin, ymax, color='tomato', linewidth=2, linestyle='--')
    plt.ylabel('Amplitude', fontsize=12) 
    plt.xlim([0, 6000])
    ymin,yma = ax.get_ylim()
    
    ax = fig.add_subplot(514) 
    plt.plot(time, p_predict, label='P-wave predicted', linestyle='--', color='orange')
    plt.legend(loc = 'upper right', borderaxespad=0., prop=legend_properties)
    plt.ylabel('Probability', fontsize=12) 
    plt.text(0.6, 0.5, f'Error={err_indx:.2f} s', fontsize=10, fontweight='bold', ha='center', va='center', transform=ax.transAxes)
    # plt.xlabel('Sample', fontsize=12) 
    plt.xlim([0, 6000])
    
    ax = fig.add_subplot(515) 
    plt.plot(time, p_labels, label='P-wave label', linestyle='--')
    plt.legend(loc = 'upper right', borderaxespad=0., prop=legend_properties)
    plt.ylabel('Probability', fontsize=12) 
    plt.xlabel('Sample', fontsize=12) 
    plt.xlim([0, 6000])
    plt.tight_layout()

def swave_signal_label_visualization(signal, s_predict, s_labels, sample_num):
    data = signal
    time = np.linspace(0, sample_num, sample_num)
    s_predict_indx = np.argmax(s_predict)
    s_labels_indx = np.argmax(s_labels)

    fig = plt.figure(figsize=(6, 6))
    ax1 = fig.add_subplot(511)
    plt.plot(time, data[:,0], 'k',label='Z')
    ymin,yma = ax1.get_ylim()
    legend_properties = {'weight':'bold'}
    ymin, ymax = ax1.get_ylim()
    plt.vlines(s_labels_indx, ymin+0.2, ymax-0.2, color='tomato', linewidth=2, label='S-arrival')
    plt.vlines(s_predict_indx, ymin, ymax, color='deepskyblue', linewidth=2, linestyle='--', label= 'S-predicted')
    plt.legend(loc = 'upper right', borderaxespad=0., prop=legend_properties)
    plt.ylabel('Amplitude', fontsize=12)
    plt.xlim([0, 6000])
    ax1.set_xticklabels([])
    
    ax = fig.add_subplot(512) 
    plt.plot(time, data[:,1], 'k',label='N')
    ymin,yma = ax1.get_ylim()
    legend_properties = {'weight':'bold'}
    ymin, ymax = ax.get_ylim()
    plt.legend(loc = 'upper right', borderaxespad=0., prop=legend_properties)
    plt.vlines(s_labels_indx, ymin+0.2, ymax-0.2, color='tomato', linewidth=2)
    plt.vlines(s_predict_indx, ymin, ymax, color='deepskyblue', linewidth=2, linestyle='--')
    plt.ylabel('Amplitude', fontsize=12) 
    plt.xlim([0, 6000])
    ax.set_xticklabels([])
    
    ax = fig.add_subplot(513) 
    plt.plot(time, data[:,2], 'k',label='E')
    ymin,yma = ax1.get_ylim()
    legend_properties = {'weight':'bold'}
    ymin, ymax = ax.get_ylim()
    plt.vlines(s_labels_indx, ymin+0.2, ymax-0.2, color='tomato', linewidth=2)
    plt.vlines(s_predict_indx, ymin, ymax, color='deepskyblue', linewidth=2, linestyle='--')
    plt.legend(loc = 'upper right', borderaxespad=0., prop=legend_properties)
    plt.ylabel('Amplitude', fontsize=12) 
    plt.xlim([0, 6000])
    ymin,yma = ax.get_ylim()
    
    ax = fig.add_subplot(514) 
    plt.plot(time, s_predict, label='S-wave predicted', linestyle='--', color='orange')
    plt.legend(loc = 'upper right', borderaxespad=0., prop=legend_properties)
    plt.ylabel('Probability', fontsize=12) 
    plt.text(0.6, 0.5, f'Error={err_indx:.2f} s', fontsize=10, fontweight='bold', ha='center', va='center', transform=ax.transAxes)
    plt.xlim([0, 6000])
    
    ax = fig.add_subplot(515) 
    plt.plot(time, s_labels, label='S-wave label', linestyle='--')
    plt.legend(loc = 'upper right', borderaxespad=0., prop=legend_properties)
    plt.ylabel('Probability', fontsize=12) 
    plt.xlabel('Sample', fontsize=12) 
    plt.xlim([0, 6000])
    plt.tight_layout()



def plot_labels_signals(signal, p_predict, s_predict, p_labels, s_labels, sample_num):
    data = signal
    p_predict_indx = np.argmax(p_predict)
    s_predict_indx = np.argmax(s_predict)
    s_labels_indx = np.argmax(s_labels)
    p_labels_indx = np.argmax(p_labels)
    
    time = np.linspace(0, sample_num, sample_num)
    fig = plt.figure(figsize=(6, 6))
    ax1 = fig.add_subplot(511)
    plt.plot(time, data[:,0], 'k',label='Z')
    ymin,ymax = ax1.get_ylim()
    # plt.vlines(spt,ymin,yma,color='r',linewidth=2)
    # plt.vlines(sst,ymin,yma,color='b',linewidth=2)
    legend_properties = {'weight':'bold'}
    # ymin, ymax = ax1.get_ylim()
    print('ymin, ymax:', ymin, ymax)
    plt.vlines(p_labels_indx,ymin+0.2,ymax-0.2,color='deepskyblue',linewidth=2)
    plt.vlines(s_labels_indx,ymin+0.2,ymax-0.2,color='tomato',linewidth=2)
    plt.vlines(p_predict_indx,ymin,ymax,color='tomato',linewidth=2, linestyle='--')
    plt.vlines(s_predict_indx,ymin,ymax,color='deepskyblue',linewidth=2, linestyle='--')
    plt.legend(loc = 'upper right', borderaxespad=0., prop=legend_properties)
    plt.ylabel('Amplitude', fontsize=12)
    plt.xlim([0, 6000])
    ax1.set_xticklabels([])
    # ax1.set_xticks([])
    
    ax = fig.add_subplot(512) 
    plt.plot(time, data[:,1], 'k',label='N')
    # ymin,ymax = ax1.get_ylim()
    plt.vlines(p_labels_indx,ymin+0.2,ymax-0.2,color='deepskyblue',linewidth=2)
    plt.vlines(s_labels_indx,ymin+0.2,ymax-0.2,color='tomato',linewidth=2)
    plt.vlines(p_predict_indx,ymin,ymax,color='tomato',linewidth=2, linestyle='--')
    plt.vlines(s_predict_indx,ymin,ymax,color='deepskyblue',linewidth=2, linestyle='--')
    legend_properties = {'weight':'bold'}
    plt.legend(loc = 'upper right', borderaxespad=0., prop=legend_properties)
    plt.ylabel('Amplitude', fontsize=12) 
    plt.xlim([0, 6000])
    ax.set_xticklabels([])
    
    ax = fig.add_subplot(513) 
    plt.plot(time, data[:,2], 'k',label='E')
    # ymin,yma = ax1.get_ylim()
    # plt.vlines(spt,ymin,yma,color='r',linewidth=2)
    # plt.vlines(sst,ymin,yma,color='b',linewidth=2)
    legend_properties = {'weight':'bold'}
    # ymin, ymax = ax.get_ylim()
    plt.vlines(p_labels_indx,ymin+0.2,ymax-0.2,color='deepskyblue',linewidth=2)
    plt.vlines(s_labels_indx,ymin+0.2,ymax-0.2,color='tomato',linewidth=2)
    plt.vlines(p_predict_indx,ymin,ymax,color='tomato',linewidth=2, linestyle='--')
    plt.vlines(s_predict_indx,ymin,ymax,color='deepskyblue',linewidth=2, linestyle='--')
    plt.legend(loc = 'upper right', borderaxespad=0., prop=legend_properties)
    plt.ylabel('Amplitude', fontsize=12) 
    plt.xlim([0, 6000])
    # plt.ylim([-1, 1])
    ymin,yma = ax.get_ylim()
    ax.set_xticklabels([])
    # ax.set_xticks([])
    
    ax = fig.add_subplot(514) 
    plt.plot(time, p_predict, label='P-wave predicted', linestyle='--')
    plt.plot(time, s_predict, label='S-wave predicted', linestyle='--')
    plt.legend(loc = 'upper right', borderaxespad=0., prop=legend_properties)
    plt.ylabel('Probability', fontsize=12) 
    # plt.xlabel('Sample', fontsize=12) 
    plt.xlim([0, 6000])
    # plt.ylim([-1, 1])
    ax.set_xticklabels([])
    
    ax = fig.add_subplot(515) 
    plt.plot(time, p_labels, label='P-wave label', linestyle='--')
    plt.plot(time, s_labels, label='S-wave label', linestyle='--')
    plt.legend(loc = 'upper right', borderaxespad=0., prop=legend_properties)
    plt.ylabel('Probability', fontsize=12) 
    plt.xlabel('Sample', fontsize=12) 
    plt.xlim([0, 6000])
    plt.tight_layout()
