import numpy as np

import tensorflow as tf
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn import metrics
from tqdm.auto import tqdm, trange
import math

def create_bi_img(img, threshold):
    '''
    Create a binary image, with values either 0 or 1 of an image with values of floats.
    --------------------------------------------------------------------------------
    img: array
        image that should be transformed to binary result
    threshold: float
        values <= threshold will be set to 0, values>threshold will be set to 1
    ---------------------------------------------------------------------------------
    returns: bi_img of same shape of input img, but values are either 0 or 1
    '''
    th_arr = np.full(shape=img.shape, fill_value=threshold)
    zero_arr = np.zeros(img.shape)
    one_arr = np.ones(img.shape)
    # create new array with where condition.Keeping nan values by using the fact that multiplying nan returns nan
    bi_img = np.where(img > th_arr, one_arr, zero_arr*img)
    return bi_img

def sample_set_wet_dry(share, x_set, x_set_wet, y_set, y_set_wet):
    idx = np.random.randint(x_set.shape[0]-x_set_wet.shape[0], size=int(share*(x_set.shape[0]-x_set_wet.shape[0])))
    x_set = x_set[idx]
    y_set = y_set[idx]
    x_set_wet_dry = np.concatenate([x_set, x_set_wet], axis=0)
    y_set_wet_dry = np.concatenate([y_set, y_set_wet], axis=0)
    return x_set_wet_dry, y_set_wet_dry

def mcc_over_time(y_true, y_pred):
    '''
    Calculates MCC score for each timestep.
    ------------------------------------------------------------------------------
    y_true: np.array
            true binary values of shape (timesteps, y, x)
    y_pred: np.array
            predicted binary values of same shape as y_true
    -----------------------------------------------------------------------------
    returns: 1D array off MCC scores for each timestep
    '''
    mcc_over_time = []
    for i in tqdm(range(0, y_true.shape[0])):
        true_flat = y_true[i].flatten()
        pred_flat = y_pred[i].flatten()
        true_clean = true_flat[~np.isnan(true_flat)]
        pred_clean = pred_flat[~np.isnan(true_flat)]
        mcc = metrics.matthews_corrcoef(true_clean, pred_clean)
        mcc_over_time.append(mcc)
    mcc_over_time = np.asarray(mcc_over_time)
    return mcc_over_time

def sel_day_fun(start, end, ds_patch):
    '''
    Only selects patches between 6AM and 6PM and returns dataset of patches only during day time.
    '''
    day_range= pd.date_range(start, end, freq='D')
    day_range
    day_list = []
    for i in range(0, len(day_range)-1):
        #day_list.append( pd.date_range(str(day_range[i])+'T02:45:00.000000000', str(day_range[i])+'T19:00:00.000000000', freq='15min'))
        day_list.append( pd.date_range(str(day_range[i])+'T06:00:00.000000000', str(day_range[i])+'T18:00:00.000000000', freq='15min'))
    time_day = pd.DatetimeIndex(np.unique(np.hstack(day_list)))
    date_lis = []
    date_lis = [np.datetime64(time_day[i]) for i in range(0, len(time_day))]
    #for i in range(0, 22):
    size = ds_patch.ID.values.size
    l = []
    for i in tqdm(range(0, size)):
        if ds_patch.times.sel(ID=i).values[0] in date_lis:
            continue
        else:
            l.append(i)
    ds_patch_day = ds_patch.drop_sel(ID=l)
    return ds_patch_day

# split ratio
def split_data(ds_seviri, ds_yw, train=0.8, val=0.1, test=0.1):
    '''
    Splits input and target datasets along time axis into training, validation and test subset
    ---------
    train, val, test: float
        each desired share. Must sum up to 1.
    ds_seviri: xarray of seviri data
        seviri satellite input data
    ds_yw: xarray of radar data
        radar target data
    ---------
    Returns
    ---------
    splitted subsets. Each three subsets for input (seviri) and target (radar)
    
    '''
    size = ds_seviri.time.values.size
    # split dataset
    if train+val+test==1:
        train_frame = [ds_seviri.time.values[0], ds_seviri.time.values[int(size*train)]]
        val_frame = [ds_seviri.time.values[int(size*train)+1], ds_seviri.time.values[int(size*train+size*val)]]
        test_frame = [ds_seviri.time.values[int(size*train+size*val)+1], ds_seviri.time.values[-1]]
        print(train_frame, val_frame, test_frame)
        # test_frame
        ds_seviri_train = ds_seviri.sel(time=slice(train_frame[0], train_frame[1]))
        ds_seviri_val = ds_seviri.sel(time=slice(val_frame[0], val_frame[1]))
        ds_yw_train = ds_yw.sel(time=slice(train_frame[0], train_frame[1]))
        ds_yw_val = ds_yw.sel(time=slice(val_frame[0], val_frame[1]))
        ds_seviri_test = ds_seviri.sel(time=slice(test_frame[0], test_frame[1]))
        ds_yw_test = ds_yw.sel(time=slice(test_frame[0], test_frame[1]))
    return ds_seviri_train, ds_seviri_val, ds_seviri_test, ds_yw_train, ds_yw_val, ds_yw_test

def normalize_test(ds_input, min_temp, max_temp, min_bright, max_bright):
    '''
    Normalize test data channelwise per Temperature channels and Brightness channels.
    -------
    ds_input: xarray dataset of inputs
    -------
    Returns
    -------
    Normalized Numpy Array of Shape (Images, Width, Height, Channels)
    '''
    array =np.stack([ np.asarray(ds_input.IR_016), 
                                np.asarray(ds_input.IR_039),
                                np.asarray(ds_input.IR_087),
                                np.asarray(ds_input.IR_108),
                                np.asarray(ds_input.IR_120),
                                np.asarray(ds_input.VIS006),
                                np.asarray(ds_input.WV_062),
                                np.asarray(ds_input.WV_073)], axis=3)
                               
    
    #array = array.reshape(array.shape[1], array.shape[2], array.shape[3], array.shape[0])
    print(array.shape)
    plt.pcolormesh(array[1, :, :, 1])
    # temperature
    np.asarray(array)[:,:,:,(1,2,3,4,6,7)] = (np.asarray(array)[:,:,:,(1,2,3,4,6,7)]- min_temp)/(max_temp - min_temp)
    # brightness
    np.asarray(array)[:,:,:,(0,5)] = (np.asarray(array)[:,:,:,(0,5)]- min_bright)/(max_bright- min_bright)
    return array

def normalize_train_val(ds_input, min_temp, max_temp, min_bright, max_bright):
    '''
    Normalize training and validation data (patches) channelwise per Temperature channels and Brightness channels.
    -------
    ds_input: xarray dataset.input
    -------
    Returns
    -------
    Normalized Numpy Array of Shape (Images, Width, Height, Channels)
    '''
    # temperature
    np.asarray(ds_input)[:,:,:,(1,2,3,4,6,7)] = (np.asarray(ds_input)[:,:,:,(1,2,3,4,6,7)]- min_temp)/(max_temp - min_temp)
    # brightness
    np.asarray(ds_input)[:,:,:,(0,5)] = (np.asarray(ds_input)[:,:,:,(0,5)]- min_bright)/(max_bright- min_bright)
    return ds_input

def to_tensor(x):
    '''
    Covert numpy array to tensor, so that each channel can be passed to the model
    -------
    x: numpy array of shape (number of patches/timesteps, width, height, channels)
        x_test, x_train, or x_val
    ------
    Returns
    ------
    each channel as input in tensor format for DL model
    '''
    x_0 = tf.convert_to_tensor(x[:, :, :, 0].reshape(x.shape[0], x.shape[1], x.shape[2], 1))#, dtype=tf.float32) #ir016
    x_1 = tf.convert_to_tensor(x[:, :, :, 1].reshape(x.shape[0], x.shape[1], x.shape[2], 1))#, dtype=tf.float32) #ir039
    x_2 = tf.convert_to_tensor(x[:, :, :, 2].reshape(x.shape[0], x.shape[1], x.shape[2], 1))#, dtype=tf.float32) #ir087
    x_3 = tf.convert_to_tensor(x[:, :, :, 3].reshape(x.shape[0], x.shape[1], x.shape[2], 1))#, dtype=tf.float32) #ir108
    x_4 = tf.convert_to_tensor(x[:, :, :, 4].reshape(x.shape[0], x.shape[1], x.shape[2], 1))#, dtype=tf.float32) #ir12
    x_5 = tf.convert_to_tensor(x[:, :, :, 5].reshape(x.shape[0], x.shape[1], x.shape[2], 1))#, dtype=tf.float32) #vis006
    x_6 = tf.convert_to_tensor(x[:, :, :, 6].reshape(x.shape[0], x.shape[1], x.shape[2], 1))#, dtype=tf.float32) #wv062
    x_7 = tf.convert_to_tensor(x[:, :, :, 7].reshape(x.shape[0], x.shape[1], x.shape[2], 1))#, dtype=tf.float32) #wv073
    return x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7

def reduce_low_rain(img, bi_img, threshold):
    return img[np.sum(bi_img, axis=(1,2)) > threshold]


def sample_set_wet_dry(share, x_set, x_set_wet, y_set, y_set_wet):
    idx = np.random.randint(x_set.shape[0]-x_set_wet.shape[0], size=int(share*(x_set.shape[0]-x_set_wet.shape[0])))
    x_set = x_set[idx]
    y_set = y_set[idx]
    x_set_wet_dry = np.concatenate([x_set, x_set_wet], axis=0)
    y_set_wet_dry = np.concatenate([y_set, y_set_wet], axis=0)
    return x_set_wet_dry, y_set_wet_dry

def downsampling(ds, th=0.05):
    '''
    Creates downsampled (/6) binary targetset out of netCDF patch dataset
    ----------
    ds: xarray patch dataset with input and target
            dataset that includes input as variable and ID as patch dimension
    ---------
    Returns
    ---------
    Numpy Array (Patches, WidthPatch, HeightPatch)
    Downsampled from 54*54 to 9*9 Patches
    '''
    shape = ds.input.shape
    target_low = np.zeros((shape[0], shape[1], shape[2]))
    for ID in tqdm(range(0, ds.ID.shape[0])):
        for ix in range(0,  9):
            for iy in range(0, 9):
                if np.any(ds.isel(ID=ID).target[iy*6:(iy+1)*6, ix*6:(ix+1)*6, 0]>=th):
                    target_low[ID, iy, ix] =1
                else:
                    target_low[ID, iy, ix] =0
    print(target_low[0])
    return target_low


def measure_distance_deg_to_m(lat1, lon1, lat2, lon2):
    
    R = 6378.137 # Radius of earth in KM
    dLat = lat2 * math.pi / 180 - lat1 * math.pi / 180
    dLon = lon2 * math.pi / 180 - lon1 * math.pi / 180
    a = math.sin(dLat/2) * math.sin(dLat/2) + math.cos(lat1 * math.pi / 180) * math.cos(lat2 * math.pi / 180) *math.sin(dLon/2) * math.sin(dLon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = R * c
    return d  # returns in km




#############################################################################################
# Loss functions

def binary_focal_loss(gamma=2., alpha=.25):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        y_true = tf.cast(y_true, tf.float32)
        # Define epsilon so that the back-propagation will not result in NaN for 0 divisor case
        epsilon = K.epsilon()
        # Add the epsilon to prediction value
        # y_pred = y_pred + epsilon
        # Clip the prediciton value
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        # Calculate p_t
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        # Calculate alpha_t
        alpha_factor = K.ones_like(y_true) * alpha
        alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        # Calculate cross entropy
        cross_entropy = -K.log(p_t)
        weight = alpha_t * K.pow((1 - p_t), gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = K.mean(K.sum(loss, axis=1))
        return loss

    return binary_focal_loss_fixed

#Tensorflow / Keras
def FocalTverskyLoss(y_true, y_pred, smooth=1e-6):
        
    '''
    Focal Tversky Loss
    '''
    if y_pred.shape[-1] <= 1:
        alpha = 0.3
        beta = 0.7
        gamma = 4/3 #5.
        y_pred = tf.keras.activations.sigmoid(y_pred)
        #y_true = y_true[:,:,:,0:1]
    elif y_pred.shape[-1] >= 2:
        alpha = 0.3
        beta = 0.7
        gamma = 4/3 #3.
        y_pred = tf.keras.activations.softmax(y_pred, axis=-1)
        y_true = K.squeeze(y_true, 3)
        y_true = tf.cast(y_true, "int32")
        y_true = tf.one_hot(y_true, num_class, axis=-1)

        
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    #flatten label and prediction tensors
    inputs = K.flatten(y_pred)
    targets = K.flatten(y_true)

    #True Positives, False Positives & False Negatives
    TP = K.sum((inputs * targets))
    FP = K.sum(((1-targets) * inputs))
    FN = K.sum((targets * (1-inputs)))

    Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
    FocalTversky = K.pow((1 - Tversky), gamma)
        
    return FocalTversky
    
def create_weighted_binary_crossentropy(zero_weight, one_weight):
    '''
    Weighted Binary Crossentropy Loss Function
    '''

    def weighted_binary_crossentropy(y_true, y_pred):

        # Original binary crossentropy (see losses.py):
        # K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

        # Calculate the binary crossentropy
        b_ce = K.binary_crossentropy(y_true, y_pred)

        # Apply the weights
        weight_vector = y_true * one_weight + (1. - y_true) * zero_weight
        weighted_b_ce = weight_vector * b_ce

        # Return the mean error
        return K.mean(weighted_b_ce)

    return weighted_binary_crossentropy

def DiceBCELoss(smooth=1e-6):  
    '''
    Dice Binary Cross Entropy Loss
    -----
    Source: https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook#Jaccard/Intersection-over-Union-(IoU)-Loss
    # similarly used in AIAI competition: https://github.com/iarai/weather4cast-2022/blob/main/models/unet_lightning.py
    '''
    def diceBCEloss(targets, inputs):
        #flatten label and prediction tensors
        inputs = K.flatten(inputs)
        targets = K.flatten(targets)

        BCE =  binary_crossentropy(targets, inputs)
        intersection = K.sum(K.dot(targets, inputs))    
        dice_loss = 1 - (2*intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
        Dice_BCE = BCE + dice_loss

        return Dice_BCE

##############################################################################
# metrics for scores
import keras.backend as K
def get_f1(y_true, y_pred): #taken from old keras source code
    '''
    Calc F1 Score for DL Training (Keras) of y_pred, where y_pred is 0 or 1.
    -------
    y_true : array of shape (n_samples,)
        True targets.
    y_pred : array of shape (n_samples,)
        Predictions of model (0/1)
    Returns
    -------
    F1 score: float
    -------
    Source: https://aakashgoel12.medium.com/how-to-add-user-defined-function-get-f1-score-in-keras-metrics-3013f979ce0d
    '''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


def matthews_correlation(y_true, y_pred):
    '''
    Calc MCC score for DL training (KERAS) of y_pred, where y_pred is round to closest int(0/1)
        -------
    y_true : array of shape (n_samples,)
        True targets.
    y_pred : array of shape (n_samples,)
        Predictions of model (0/1)
    Returns
    -------
    MCC score: float
    '''
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())


def false_neg_rate(y_true, y_pred):
    '''
    Calc FNR score for DL training of y_pred, where y_pred is round to closest int(0/1)
        -------
    y_true : array of shape (n_samples,)
        True targets.
    y_pred : array of shape (n_samples,)
        Predictions of model (0/1)
    Returns
    -------
    FNR score: float
    '''
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    denominator = (tp + fn)
    return fn / denominator


def true_neg_rate(y_true, y_pred):
    '''
    Calc TNR score for DL training of y_pred, where y_pred is round to closest int(0/1)
    -------
    y_true : array of shape (n_samples,)
        True targets.
    y_pred : array of shape (n_samples,)
        Predictions of model (0/1)
    Returns
    -------
    TNR score: float
    
    '''
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)
    
    denominator = (fp + tn)
    return tn / denominator

# Brier-score
from sklearn.metrics import brier_score_loss
def brier_score(y_true, y_pred):
    # https://github.com/scikit-learn/scikit-learn/blob/36958fb24/sklearn/metrics/_classification.py#L2614
    """Compute the Brier score loss.
    The smaller the Brier score loss, the better, hence the naming with "loss".
    The Brier score measures the mean squared difference between the predicted
    probability and the actual outcome. The Brier score always
    takes on a value between zero and one, since this is the largest
    possible difference between a predicted probability (which must be
    between zero and one) and the actual outcome (which can take on values
    of only 0 and 1). It can be decomposed is the sum of refinement loss and
    calibration loss.
    The Brier score is appropriate for binary and categorical outcomes that
    can be structured as true or false, but is inappropriate for ordinal
    variables which can take on three or more values (this is because the
    Brier score assumes that all possible outcomes are equivalently
    "distant" from one another). Which label is considered to be the positive
    label is controlled via the parameter `pos_label`, which defaults to
    the greater label unless `y_true` is all 0 or all -1, in which case
    `pos_label` defaults to 1.
    Read more in the :ref:`User Guide <brier_score_loss>`.
    Parameters
    ----------
    y_true : array of shape (n_samples,)
        True targets.
    y_prob : array of shape (n_samples,)
        Probabilities of the positive class.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    pos_label : int or str, default=None
        Label of the positive class. `pos_label` will be inferred in the
        following manner:
        * if `y_true` in {-1, 1} or {0, 1}, `pos_label` defaults to 1;
        * else if `y_true` contains string, an error will be raised and
          `pos_label` should be explicitly specified;
        * otherwise, `pos_label` defaults to the greater label,
          i.e. `np.unique(y_true)[-1]`.
    Returns
    -------
    score : float
        Brier score loss.
    References
    ----------
    .. [1] `Wikipedia entry for the Brier score
            <https://en.wikipedia.org/wiki/Brier_score>`_.

    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    return brier_score_loss(y_true, y_pred, sample_weight=None, pos_label=1)