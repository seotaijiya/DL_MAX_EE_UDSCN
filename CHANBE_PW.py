from __future__ import absolute_import, division, print_function

#import tensorflow.compat.v1 as tf
#tf.compat.v1.disable_eager_execution()

import tensorflow as tf


import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from itertools import product
np.set_printoptions(precision=5)
import time
import itertools
from tensorflow.keras import backend as K




'''
    Find the feasible location of users
    This function finds the feasible location of INITIALIZATION
    In this function, the location of RX is randomly chosen such that its distance with
    TX is within Dist_TX_RX
'''


def Feasible_Loc_Init(Cur_loc, Size_area, Dist_TX_RX):
    temp_dist = Dist_TX_RX * (np.random.rand(1, 2) - 0.5) * 2
    temp_chan = Cur_loc + temp_dist
    while (np.max(abs(temp_chan)) > Size_area / 2) | (np.linalg.norm(temp_dist) > Dist_TX_RX):
        temp_dist = Dist_TX_RX * (np.random.rand(1, 2) - 0.5) * 2
        temp_chan = Cur_loc + temp_dist
    return temp_chan




'''
    Initialization of location information.
    It will return the location of one PU and SUs.
    The Users will be allocated to 2D area whose range is -Size_area/2 ~ Size_area/2.
    The distance between RX and TX for the same transmit pair is limited to "Dist_TX_RX"

    Input: Size_area, Dist_TX_RX, Num_D2D, Num_Ch
    -> Number of channel is same with the number of CUE

'''


def loc_init(Size_area, Dist_TX_RX, Num_BS, Num_MS):
    tx_loc = Size_area * (np.random.rand(Num_BS, 2) - 0.5)
    rx_loc = Size_area * (np.random.rand(Num_MS, 2) - 0.5)
    return rx_loc, tx_loc



'''
    Determine the channel gain  (UPLINK)

    --------------------------------------------
    ** The basic setting
      : Pathloss exponent -> 3.8
      : Pathloss constant -> 34.5
      : Moving speed      -> 3km/h
      : Power update      -> 100ms
      : Reset location    -> every 100 samples
    --------------------------------------------


    Each channel is time varying according to speed

    Location of all users are initialized every 1000 samples.

    The location of CUE for each band is different

    The output looks like as follows:

    output[Sample][Channel][Users][Users]


    Example for 2 D2D and 1 CUE channel   :
    [
        h_{D2D_1 -> D2D_1}        h_{D2D_1->D2D_2}          h_{D2D_1->BS}
        h_{D2D_2 -> D2D_1}        h_{D2D_2->D2D_2}          h_{D2D_2->CUE}
        h_{CUE -> D2D_1}          h_{CUE->D2D_2}            h_{CUE->BS}
    ]


    Accordingly, output[0] returns the first sample
    output[0][0] return the (N+1) X (N+1) channel gain for first band

'''


def ch_gen(Size_area, D2D_dist, Num_BS, Num_MS, Num_samples, PL_alpha=38., PL_const=34.5):
    ch_w_fading = []


    ## Calculate the
    for i in range(Num_samples):
        rx_loc, tx_loc = loc_init(Size_area, D2D_dist, Num_BS, Num_MS)
        tx_loc_with_CUE = tx_loc
        ## generate distance_vector
        dist_vec = rx_loc.reshape(Num_MS, 1, 2) - tx_loc_with_CUE
        dist_vec = np.linalg.norm(dist_vec, axis=2)
        dist_vec = np.maximum(dist_vec, 5)

        # find path loss // shadowing is not considered
        pu_ch_gain_db = - PL_const - PL_alpha * np.log10(dist_vec)
        pu_ch_gain = 10 ** (pu_ch_gain_db / 10)
        multi_fading = 0.5 * np.random.randn(Num_MS , Num_BS ) ** 2 + 0.5 * np.random.randn(Num_MS, Num_BS) ** 2
        final_ch = np.maximum(pu_ch_gain * multi_fading, np.exp(-30))
        ch_w_fading.append(np.transpose(final_ch))


    return np.array(ch_w_fading)


'''
Find the optimal value for one sample with one channel

The shape of input channel is as follows:

    -> channel[users][users]

Note that i

    Example for 2 D2D and 1 CUE channel   :
    [
        h_{D2D_1 -> D2D_1}        h_{D2D_1->D2D_2}          h_{D2D_1->BS}
        h_{D2D_2 -> D2D_1}        h_{D2D_2->D2D_2}          h_{D2D_2->BS}
        h_{CUE -> D2D_1}          h_{CUE->D2D_2}            h_{CUE->BS}
    ]


'''






def cal_SINR_one_sample_one_channel(channel, tx_power, channel_assign_vec, int_assign_vec, noise, P_c_A, P_c_S):
    ## Note that we transpose the channel to
    tot_ch = np.multiply(channel, np.expand_dims(tx_power, -1))
    sig_ch = np.sum(tot_ch * channel_assign_vec, axis=1)
    int_ch = np.sum(tot_ch * int_assign_vec, axis=1)

    power_val = np.sum(np.multiply(np.expand_dims(tx_power, -1), channel_assign_vec), axis=1)

    power_active = np.sum(channel_assign_vec, axis=1)
    power_sleep = 1-power_active

    power_active = power_active * P_c_A
    power_sleep = power_sleep * P_c_S



    SINR_val = np.divide(sig_ch, int_ch+noise)
    cap_val = np.log(1.0+SINR_val) * 1.4427

    EE_val = np.divide(np.sum(cap_val, axis=1), np.sum(power_val, axis=1) + np.sum(power_active) + np.sum(power_sleep))

    return EE_val, cap_val




def cal_SINR_one_sample_one_channel_NP(channel, tx_power, channel_assign_vec, int_assign_vec, noise, P_c_A, P_c_S):
    ## Note that we transpose the channel to


    tot_ch = np.multiply(channel, np.expand_dims(tx_power, -1))
    sig_ch = np.sum(tot_ch * channel_assign_vec, axis=0)
    int_ch = np.sum(tot_ch * int_assign_vec, axis=0)
    power_val = np.sum(np.multiply(np.expand_dims(tx_power, -1), channel_assign_vec), axis=0)
    SINR_val = np.divide(sig_ch, int_ch+noise)
    cap_val = np.log(1.0+SINR_val) * 1.4427

    power_active = np.sum(channel_assign_vec, axis=1)
    power_sleep = 1-power_active

    power_active = power_active * P_c_A
    power_sleep = power_sleep * P_c_S


    EE_val = np.divide(np.sum(cap_val, axis=0), np.sum(power_val, axis=0) + np.sum(power_active) + np.sum(power_sleep))

    return EE_val, cap_val










def cal_rate_NP(channel, TX_mat, Assign_mat, noise, P_c_A, P_c_S):

    ## IMPORTANT: We asusme that num BS < Num MS
    num_samples = channel.shape[0]
    num_BS = channel.shape[1]
    num_MS = channel.shape[2]

    tot_EE = 0
    tot_SE = 0
    power_mat = []
    chan_assign_mat = []

    for i in range(num_samples):
        tx_power = TX_mat[i]

        max_EE = 0
        max_SE = 0
        cur_ch = channel[i]
        channel_assign_vec = np.zeros((num_BS, num_MS))
        int_assign_vec = np.zeros((num_BS, num_MS))
        for j in range(num_BS):
            MS_sel = np.argmax(Assign_mat[i, j])
            if np.max(Assign_mat[i, j]) > 0.5:
                int_assign_vec[:, MS_sel] = 1
                int_assign_vec[j, MS_sel] = 0
                channel_assign_vec[j, MS_sel] = 1


        cur_ch_ee, cur_ch_cap = cal_SINR_one_sample_one_channel_NP(cur_ch, tx_power, channel_assign_vec, int_assign_vec, noise, P_c_A, P_c_S)



        cur_EE_sum = np.sum(cur_ch_ee)
        cur_SE_sum = np.sum(cur_ch_cap)

        tot_EE = tot_EE + cur_EE_sum
        tot_SE = tot_SE + cur_SE_sum


    tot_EE = tot_EE / num_samples
    tot_SE = tot_SE / num_samples


    return tot_EE, tot_SE















'''
    Find all possible combination
    Use product to find combination
'''


def all_possible_tx_power(num_user, granuty):

    items = [np.arange(granuty)] * num_user
    temp_power = list(itertools.product(*items))
    temp_power = np.reshape(temp_power, (-1, num_user))

    temp_power_1 = np.reshape(temp_power, (-1, num_user))
    power = np.reshape(temp_power, (-1, num_user)) / (granuty - 1)

    power_mat = []
    for i in range(power.shape[0]):
        sum_val = np.sum(power[i])
        if sum_val != 0:
            power_mat.append(power[i])
    return np.array(power_mat)



'''
Find all possible combination

How to use it?

>> optimal_power(channel, granuity of transmit power, noise, CUE_thr)

CUE_thr: Determine the minimum data rate required for CUE
DUE_thr: Determine the minimum data rate required for D2D user

Returned value is the sum of all D2D users AND POWER val


'''



def optimal_power(channel, tx_max, granuty, noise, P_c_A, P_c_S, tx_power_set):

    ## IMPORTANT: We asusme that num BS < Num MS
    num_samples = channel.shape[0]
    num_BS = channel.shape[1]
    num_MS = channel.shape[2]

    tot_EE = 0
    tot_SE = 0
    power_mat = []
    chan_assign_mat = []

    for i in range(num_samples):
        tx_power = tx_power_set * tx_max

        max_EE = 0
        max_SE = 0
        cur_ch = channel[i]
        for i_1 in range(num_MS+1):
            for i_2 in range(num_MS+1):
                for i_3 in range(num_MS+1):
                    channel_assign_vec = np.zeros((num_BS, num_MS))
                    int_assign_vec = np.zeros((num_BS, num_MS))

                    if i_1 != 0:
                        channel_assign_vec[0, i_1-1] = 1
                    if (i_2 != 0) & (i_2 != i_1):
                        channel_assign_vec[1, i_2-1] = 1
                    if (i_3 != 0) & (i_3 != i_1) & (i_3 != i_2):
                        channel_assign_vec[2, i_3 - 1] = 1

                    for iii in range(num_MS):
                        temp_index = np.where(channel_assign_vec[:,iii] == 1)
                        int_assign_vec[:, iii] = 1
                        int_assign_vec[temp_index, iii] = 0

                    cur_ch_ee, cur_ch_cap = cal_SINR_one_sample_one_channel(cur_ch, tx_power, channel_assign_vec, int_assign_vec, noise, P_c_A, P_c_S)

                    cur_EE_sum = cur_ch_ee
                    cur_EE_sum_max = np.max(cur_EE_sum)

                    if cur_EE_sum_max > max_EE:
                        EE_max_arg = np.argmax(cur_EE_sum)
                        max_EE = cur_EE_sum_max
                        sum_cur_ch_cap = np.sum(cur_ch_cap, axis=1)
                        max_SE = sum_cur_ch_cap[EE_max_arg]
                        max_tx_power_cur = tx_power[EE_max_arg]/tx_max
                        max_cha_assig_cur = channel_assign_vec

        tot_EE = tot_EE + max_EE
        tot_SE = tot_SE + max_SE
        power_mat.append(max_tx_power_cur)
        chan_assign_mat.append(max_cha_assig_cur)


    tot_EE = tot_EE / num_samples
    tot_SE = tot_SE / num_samples

    return tot_EE, tot_SE, np.array(power_mat), np.array(chan_assign_mat)




def gr_BI_power(channel, tx_max, granuty, noise, P_c_A, P_c_S, tx_power_set):

    ## IMPORTANT: We asusme that num BS < Num MS
    num_samples = channel.shape[0]
    num_BS = channel.shape[1]
    num_MS = channel.shape[2]

    tot_EE = 0
    tot_SE = 0
    power_mat = []
    chan_assign_mat = []

    for i in range(num_samples):
        tx_power = tx_power_set * tx_max

        max_EE = 0
        max_SE = 0
        cur_ch = channel[i]
        channel_assign_vec = np.zeros((num_BS, num_MS))
        int_assign_vec = np.zeros((num_BS, num_MS))

        argmaxchan = np.copy(cur_ch)
        for iii in range(num_BS):
            arg_val = np.argmax(argmaxchan[iii, :], axis=0)
            channel_assign_vec[iii, arg_val] = 1
            argmaxchan[:,arg_val] = -1000

        for iii in range(num_MS):
            temp_index = np.where(channel_assign_vec[:,iii] == 1)
            int_assign_vec[:, iii] = 1
            int_assign_vec[temp_index, iii] = 0

        cur_ch_ee, cur_ch_cap = cal_SINR_one_sample_one_channel(cur_ch, tx_power, channel_assign_vec, int_assign_vec, noise, P_c_A, P_c_S)

        cur_EE_sum = cur_ch_ee
        cur_EE_sum_max = np.max(cur_EE_sum)

        EE_max_arg = np.argmax(cur_EE_sum)
        max_EE = cur_EE_sum_max
        sum_cur_ch_cap = np.sum(cur_ch_cap, axis=1)
        max_SE = sum_cur_ch_cap[EE_max_arg]
        max_tx_power_cur = tx_power[EE_max_arg]/tx_max
        max_cha_assig_cur = channel_assign_vec

        tot_EE = tot_EE + max_EE
        tot_SE = tot_SE + max_SE
        power_mat.append(max_tx_power_cur)
        chan_assign_mat.append(max_cha_assig_cur)


    tot_EE = tot_EE / num_samples
    tot_SE = tot_SE / num_samples

    return tot_EE, tot_SE, np.array(power_mat), np.array(chan_assign_mat)




def gr_MAX_power(channel, tx_max, granuty, noise, P_c_A, P_c_S, tx_power_set_not_used):

    ## IMPORTANT: We asusme that num BS < Num MS
    num_samples = channel.shape[0]
    num_BS = channel.shape[1]
    num_MS = channel.shape[2]

    tot_EE = 0
    tot_SE = 0
    power_mat = []
    chan_assign_mat = []

    for i in range(num_samples):
        tx_power = np.ones((2, Num_BS)) * tx_max

        max_EE = 0
        max_SE = 0
        cur_ch = channel[i]
        channel_assign_vec = np.zeros((num_BS, num_MS))
        int_assign_vec = np.zeros((num_BS, num_MS))

        argmaxchan = np.copy(cur_ch)
        for iii in range(num_BS):
            arg_val = np.argmax(argmaxchan[iii, :], axis=0)
            channel_assign_vec[iii, arg_val] = 1
            argmaxchan[:,arg_val] = -1000

        for iii in range(num_MS):
            temp_index = np.where(channel_assign_vec[:,iii] == 1)
            int_assign_vec[:, iii] = 1
            int_assign_vec[temp_index, iii] = 0

        cur_ch_ee, cur_ch_cap = cal_SINR_one_sample_one_channel(cur_ch, tx_power, channel_assign_vec, int_assign_vec, noise, P_c_A, P_c_S)

        cur_EE_sum = cur_ch_ee
        cur_EE_sum_max = np.max(cur_EE_sum)

        EE_max_arg = np.argmax(cur_EE_sum)
        max_EE = cur_EE_sum_max
        sum_cur_ch_cap = np.sum(cur_ch_cap, axis=1)
        max_SE = sum_cur_ch_cap[EE_max_arg]
        max_tx_power_cur = tx_power[EE_max_arg]/tx_max
        max_cha_assig_cur = channel_assign_vec

        tot_EE = tot_EE + max_EE
        tot_SE = tot_SE + max_SE
        power_mat.append(max_tx_power_cur)
        chan_assign_mat.append(max_cha_assig_cur)


    tot_EE = tot_EE / num_samples
    tot_SE = tot_SE / num_samples

    return tot_EE, tot_SE, np.array(power_mat), np.array(chan_assign_mat)








'''

    LOSS MODELS

'''


'''
    This function calculates the capacity of D2D users and CUE userprint
    The return of this function is D2D capacity and CUE capacity
    Capacity for each channel is acculmulated.

    The shape of return is as follows:

    [Num_sample, Num_users]

    The output will be capacity of D2D and capacity of CUE

'''


def cal_EE_tf(channel, tx_pow_chan, alloc_chan, tx_max, noise, P_c_A, P_c_S, log_data_mean, log_data_std, Num_BS, Num_MS):

    alloc_chan_rev = alloc_chan
    channel_rev = tf.exp(channel * log_data_std + log_data_mean)
    inter_mat = tf.constant(1.0) - alloc_chan
    power_mat = tf.multiply(alloc_chan_rev, tx_pow_chan)


    sig_ch = tf.multiply(channel_rev, alloc_chan_rev)
    int_ch = tf.multiply(channel_rev, inter_mat)

    sig_ch = tf.multiply(sig_ch, tx_pow_chan)
    int_ch = tf.multiply(int_ch, tx_pow_chan)

    sig_ch = tf.reduce_sum(sig_ch, axis=1)
    int_ch = tf.reduce_sum(int_ch, axis=1)
    power_mat = tf.reduce_sum(power_mat, axis=1)


    BS_active = tf.reduce_sum(alloc_chan_rev, axis=2)
    power_active = tf.reduce_sum(BS_active * P_c_A, axis=1)
    power_sleep = tf.reduce_sum((tf.constant(1.0) - BS_active) * P_c_S, axis=1)

    SINR_val = tf.div(sig_ch, int_ch + noise)
    CAP_val = tf.reduce_sum(tf.log(tf.constant(1.0) + SINR_val) * tf.constant(1.4427), axis=1)

    EE_val = tf.div(CAP_val, tf.reduce_sum(power_mat, axis=1) + power_active + power_sleep) * tf.constant(1000.0)

    return EE_val, CAP_val







def cal_LOSS_Total_tf(channel, tf_output, noise, tx_max, P_c_A, P_c_S, log_data_mean, log_data_std, Num_BS, Num_MS):
    tx_pow_chan = tf.minimum(tf_output[:, :, -1:], 1.0)*tx_max
    alloc_chan = tf.minimum(tf_output[:, :, :-1], 1.0)

    EE_val, CAP_val = cal_EE_tf(channel, tx_pow_chan, alloc_chan, tx_max, noise, P_c_A, P_c_S, log_data_mean, log_data_std, Num_BS, Num_MS)

    chan_assign_temp = tf.nn.relu(tf_output[:, :, :-1] - tf.constant(1 / Num_MS))
    chan_assign_temp = tf.nn.sigmoid(chan_assign_temp * tf.constant(10.0))
    chan_assign_temp = tf.nn.relu(tf.reduce_sum(chan_assign_temp, axis=2) - tf.constant(1.0))

    integer_vio = tf.reduce_mean(tf.pow( tf.nn.relu(tf_output[:, 0, :-1] + tf_output[:, 1, :-1] + tf_output[:, 2, :-1]-tf.constant(1.0)), 4), axis=1)

    Loss = -tf.reduce_mean(EE_val)  + tf.constant(0.001) * integer_vio

    return Loss






def cal_LOSS_EE_tf(channel, tf_output, noise, tx_max, P_c_A, P_c_S, log_data_mean, log_data_std, Num_BS, Num_MS):
    tx_pow_chan = tf.minimum(tf_output[:, :, -1:], 1.0)*tx_max
    alloc_chan = tf.minimum(tf_output[:, :, :-1], 1.0)

    EE_val, CAP_val = cal_EE_tf(channel, tx_pow_chan, alloc_chan, tx_max, noise, P_c_A, P_c_S, log_data_mean, log_data_std, Num_BS, Num_MS)

    Loss = tf.reduce_mean(EE_val)
    return Loss




def cal_LOSS_SE_tf(channel, tf_output, noise, tx_max, P_c_A, P_c_S, log_data_mean, log_data_std, Num_BS, Num_MS):
    tx_pow_chan = tf.minimum(tf_output[:, :, -1:], 1.0)*tx_max
    alloc_chan = tf.minimum(tf_output[:, :, :-1], 1.0)

    EE_val, CAP_val = cal_EE_tf(channel, tx_pow_chan, alloc_chan, tx_max, noise, P_c_A, P_c_S, log_data_mean, log_data_std, Num_BS, Num_MS)

    Loss = tf.reduce_mean(CAP_val)
    return Loss








'''
    This function calculates the loss for RATE
'''

def cal_LOSS_init_tf(channel, tf_output, y_true):
    Loss = tf.reduce_mean(tf.reduce_mean(tf.pow(tf_output-y_true, 2), axis=2))
    return Loss


def Total_EE_loss_wrapper(input_tensor, noise, tx_max, P_c_A, P_c_S, log_data_mean, log_data_std, Num_BS, Num_MS):
    def TOTAL_loss(y_true, y_pred):
        Loss = cal_LOSS_Total_tf(input_tensor, y_pred, noise, tx_max, P_c_A, P_c_S, log_data_mean, log_data_std, Num_BS, Num_MS)
        return Loss
    return TOTAL_loss


def SE_loss_wrapper(input_tensor, noise, tx_max, P_c_A, P_c_S, log_data_mean, log_data_std, Num_BS, Num_MS):
    def SE_loss(y_true, y_pred):
        Loss = cal_LOSS_SE_tf(input_tensor, y_pred, noise, tx_max, P_c_A, P_c_S, log_data_mean, log_data_std, Num_BS, Num_MS)
        return Loss
    return SE_loss


def EE_loss_wrapper(input_tensor, noise, tx_max, P_c_A, P_c_S, log_data_mean, log_data_std, Num_BS, Num_MS):
    def EE_loss(y_true, y_pred):
        Loss = cal_LOSS_EE_tf(input_tensor, y_pred, noise, tx_max, P_c_A, P_c_S, log_data_mean, log_data_std, Num_BS, Num_MS)
        return Loss
    return EE_loss








'''

    DNN MODELS

'''


def DNN_basic_module(Input_layer, Num_weights_inner, Num_outputs, Num_layers=3, activation='relu'):
    Inner_layer = layers.Dense(Num_weights_inner)(Input_layer)
    Inner_layer_in = layers.Activation('relu')(Inner_layer)

    ## Number of layers should be at least 2
    assert Num_layers > 1

    for i in range(Num_layers - 2):
        Inner_layer_in = layers.Dense(Num_weights_inner)(Inner_layer_in)
        Inner_layer_in = layers.BatchNormalization()(Inner_layer_in)
        Inner_layer_in = layers.Activation('relu')(Inner_layer_in)
        Inner_layer_in = layers.Dropout(0.1)(Inner_layer_in)

    Out_layer = layers.Dense(Num_outputs)(Inner_layer_in)
    return Out_layer



"""
    Construct model with full CSI
"""


def DNN_model_full(Num_BS, Num_MS, Num_weights, Num_layers=4):
    inputs = tf.keras.Input(shape=(Num_BS, Num_MS))
    inputs_reshape = layers.Flatten(input_shape=(Num_BS, Num_MS))(inputs)

    ## Find the results for Power level
    result_PL = DNN_basic_module(inputs_reshape, Num_weights, Num_BS, Num_layers)
    result_PL = layers.Reshape((Num_BS, 1))(result_PL)
    result_PL = layers.Activation('sigmoid')(result_PL)


    ## Find the results for Resourace allocation
    result_RA_SIG = DNN_basic_module(inputs_reshape, Num_weights, Num_BS, Num_layers)
    result_RA_SIG = layers.Reshape((Num_BS, 1))(result_RA_SIG)
    result_RA_SIG = layers.Activation('sigmoid')(result_RA_SIG)

    result_RA_BS = DNN_basic_module(inputs_reshape, Num_weights, Num_BS*Num_MS, Num_layers)
    result_RA_BS = layers.Reshape((Num_BS, Num_MS))(result_RA_BS)
    result_RA_BS = layers.Activation('softmax')(result_RA_BS)

    result_RA = layers.Multiply()([result_RA_SIG, result_RA_BS])
    #result_RA = layers.Add()([result_RA_MS, result_RA_BS])

    result = layers.Concatenate()([result_RA, result_PL])

    model = tf.keras.Model(inputs=inputs, outputs=result)

    return model







####################################################################################
####################################################################################
####################################################################################
####################################################################################


Num_BS = 3
Num_MS = 3
Num_power_level = 20
Num_layers_full = 6
Num_weights_full = 400


BW = 1e7
noise = BW*10**-17.4
num_samples_tr = int(2*1e6)
num_samples_val = int(1e4)
Size_area = 1000
D2D_dist = 15
batch_size_set = 1024 * 8


epoch_num = 1000

EE_MAT_TOT = []
SE_MAT_TOT = []

#np.random.seed(0)
#tf.set_random_seed(0)



EE_MAT_TOT = []
SE_MAT_TOT = []
PW_MAT_TOT = []
AC_MAT_TOT = []




for outer_loop in range(5):

    EE_MAT_TEMP = []
    SE_MAT_TEMP = []
    PW_MAT_TEMP = []
    AC_MAT_TEMP = []


    Size_area = 200.0
    P_c_A = 10 ** (3.94)
    P_c_S = 10 ** (3.756)
    tx_max = 10 ** (2.6 + 0.5 * outer_loop)

    tx_power_set = all_possible_tx_power(Num_BS, Num_power_level - 1)
    tx_power_set_BI = all_possible_tx_power(Num_BS, 2)


    data_train = np.array(ch_gen(Size_area, D2D_dist, Num_BS, Num_MS, num_samples_tr))

    data_test = np.array(ch_gen(Size_area, D2D_dist, Num_BS, Num_MS, num_samples_val))



    EE_OPT, SE_OPT, OPT_power, OPT_chan = optimal_power(data_test, tx_max, Num_power_level, noise, P_c_A, P_c_S, tx_power_set)
    print("Opt: EE = %0.3f, SE = %0.3f, TX power = %0.3f, Active rate = %0.3f" % (1000*EE_OPT, SE_OPT, tx_max * np.mean(OPT_power), np.mean(OPT_chan)*Num_MS))
    print("")

    EE_MAT_TEMP.append(1000*EE_OPT)
    SE_MAT_TEMP.append(SE_OPT)
    PW_MAT_TEMP.append(tx_max * np.mean(OPT_power))
    AC_MAT_TEMP.append(np.mean(OPT_chan)*Num_MS)





    EE_LCOM, SE_LCOM, LCOM_power, LCOM_chan = optimal_power(data_test, tx_max, 3, noise, P_c_A, P_c_S, tx_power_set_BI)
    print("Binary-opt: EE = %0.3f, SE = %0.3f, TX power = %0.3f, Active rate = %0.3f" % (1000*EE_LCOM, SE_LCOM, tx_max * np.mean(LCOM_power), np.mean(LCOM_chan)*Num_MS))
    print("")

    EE_MAT_TEMP.append(1000*EE_LCOM)
    SE_MAT_TEMP.append(SE_LCOM)
    PW_MAT_TEMP.append(tx_max * np.mean(LCOM_power))
    AC_MAT_TEMP.append(np.mean(LCOM_chan)*Num_MS)




    EE_BIN, SE_BIN, BIN_power, BIN_chan = gr_BI_power(data_test, tx_max, 3, noise, P_c_A, P_c_S, tx_power_set_BI)
    print("Binary-greedy: EE = %0.3f, SE = %0.3f, TX power = %0.3f, Active rate = %0.3f" % (1000 * EE_BIN, SE_BIN, tx_max * np.mean(BIN_power), np.mean(BIN_chan) * Num_MS))
    print("")


    EE_MAT_TEMP.append(1000*EE_BIN)
    SE_MAT_TEMP.append(SE_BIN)
    PW_MAT_TEMP.append(tx_max * np.mean(BIN_power))
    AC_MAT_TEMP.append(np.mean(BIN_chan)*Num_MS)







    EE_MAX, SE_MAX, MAX_power, MAX_chan  = gr_MAX_power(data_test, tx_max, 3, noise, P_c_A, P_c_S, tx_power_set_BI)
    print("MAX: EE = %0.3f, SE = %0.3f, TX power = %0.3f, Active rate = %0.3f" % (1000 * EE_MAX, SE_MAX, tx_max * np.mean(MAX_power), np.mean(MAX_chan) * Num_MS))
    print("")

    EE_MAT_TEMP.append(1000*EE_MAX)
    SE_MAT_TEMP.append(SE_MAX)
    PW_MAT_TEMP.append(tx_max * np.mean(MAX_power))
    AC_MAT_TEMP.append(np.mean(MAX_chan)*Num_MS)








    ## For random case
    Tx_power_rand = tx_max *  np.random.rand(data_test.shape[0], Num_BS)
    Chan_assign_rand = np.zeros((data_test.shape[0], Num_BS, Num_BS))
    for iii in range(Num_BS):
        Chan_assign_rand[:, iii, iii] = 1

    EE_RAN, SE_RAN = cal_rate_NP(data_test, Tx_power_rand, Chan_assign_rand, noise, P_c_A, P_c_S)
    print("Random: EE = %0.3f, SE = %0.3f, TX power = %0.3f, Active rate = %0.3f" % (1000 * EE_RAN, SE_RAN, np.mean(Tx_power_rand), np.mean(Chan_assign_rand) * Num_MS))
    print("")


    EE_MAT_TEMP.append(1000*EE_RAN)
    SE_MAT_TEMP.append(SE_RAN)
    PW_MAT_TEMP.append(np.mean(Tx_power_rand))
    AC_MAT_TEMP.append(np.mean(Chan_assign_rand) * Num_MS)








    ## Recalculate the number of feasible solutions
    num_samples = data_train.shape[0]
    labels = np.zeros(num_samples, )

    log_data = np.log(data_train)
    log_data_mean = np.mean(log_data)
    log_data_std = np.std(log_data)
    log_data = (log_data - log_data_mean) / log_data_std



    log_data_test = np.log(data_test)
    log_data_test = (log_data_test - log_data_mean) / log_data_std




    print("")
    print("Outer_loop: %d "%outer_loop)

    learning_rate_cur = 2.0 * 1e-4


    cap_full = []
    cap_dist = []
    prob_full = []
    prob_dist = []

    ######################################################################################################
    model_SE = DNN_model_full(Num_BS, Num_MS, Num_weights_full, Num_layers_full)



    model_SE.compile(optimizer=tf.train.AdamOptimizer(learning_rate_cur),
                     loss=Total_EE_loss_wrapper(model_SE.input, noise, tx_max, P_c_A, P_c_S, log_data_mean, log_data_std, Num_BS, Num_MS),
                     metrics=[EE_loss_wrapper(model_SE.input, noise, tx_max, P_c_A, P_c_S, log_data_mean, log_data_std, Num_BS, Num_MS),
                              SE_loss_wrapper(model_SE.input, noise, tx_max, P_c_A, P_c_S, log_data_mean, log_data_std, Num_BS, Num_MS)]
                     )

    ######################################################################################################


    for i in range(5):
        model_SE.fit(log_data, labels, batch_size=batch_size_set, epochs=60, verbose=2)

        Tx_power_DNN = model_SE.predict(log_data_test)
        Tx_power_DNN_predict = np.squeeze(Tx_power_DNN[:, :, -1:], axis=2)

        Chan_assign_DNN_predict = Tx_power_DNN[:,:,:-1]

        EE_DL, SE_DL= cal_rate_NP(data_test, tx_max * Tx_power_DNN_predict, Chan_assign_DNN_predict, noise, P_c_A, P_c_S)

        print("PROP: EE = %0.3f, SE = %0.3f, TX power = %0.3f, Active rate = %0.3f" % (1000 * EE_DL, SE_DL, tx_max * np.mean(Tx_power_DNN_predict), np.mean(Chan_assign_DNN_predict) * Num_MS))




    EE_MAT_TEMP.append(1000*EE_DL)
    SE_MAT_TEMP.append(SE_DL)
    PW_MAT_TEMP.append(tx_max * np.mean(Tx_power_DNN_predict))
    AC_MAT_TEMP.append(np.mean(Chan_assign_DNN_predict) * Num_MS)


    EE_MAT_TOT.append(EE_MAT_TEMP)
    SE_MAT_TOT.append(SE_MAT_TEMP)
    PW_MAT_TOT.append(PW_MAT_TEMP)
    AC_MAT_TOT.append(AC_MAT_TEMP)

    print("")
    print("")
    print("*"*100)
    print("*"*100)
    print("")
    print("")


print("")
print("")
print("EE: ")
print(np.array(EE_MAT_TOT))
print("")
print("SE: ")
print(np.array(SE_MAT_TOT))
print("")
print("PW: ")
print(np.array(PW_MAT_TOT))
print("")
print("AC: ")
print(np.array(AC_MAT_TOT))