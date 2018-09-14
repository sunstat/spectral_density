import numpy as np
import os
import matplotlib.pyplot as plt


def index_to_freq(ind, n):
    return 2*np.pi*ind/n



def generate_mvar(transition_matrices, num_obs, stdev, starting_vec=None, noise_type='G'):
    lag = len(transition_matrices)
    p, _ = transition_matrices[0].shape
    if starting_vec is None:
        starting_vec = np.random.normal(0, stdev, (lag, p))
    ts = np.zeros((num_obs, p))
    ts[0:lag, :] = starting_vec
    for i in range(lag, num_obs):
        for item in range(0, lag):
            ts[i, :] += np.dot(transition_matrices[item], ts[i - item - 1, :])
        if noise_type == 'G':
            errs = np.random.normal(0, stdev, (p,))
        elif noise_type == 'T':
            errs = np.random.standard_t(5, (p,))
        ts[i, :] += errs
    return ts



def ma_help(weights, noise_arr, stdev):
    '''
    :param weights: weights for the noise
    :param noise_arr: noise_array, during each iteration,
    we delete the first one and append a new generated error vector
    :param stdev:
    :return:
    '''
    length = len(weights)
    p, _ = weights[0].shape
    value = 0
    for i in range(length-1, -1, -1):
        value += np.dot(weights[length-i-1], noise_arr[i])
    noise_arr.pop(0)
    noise_arr.append(np.random.normal(0, stdev, (p,)))
    return value



def generate_ma(weight_matrices, num_obs, stdev, noise_type='G'):
    length = len(weight_matrices)
    p, _ = weight_matrices[0].shape
    if noise_type == 'G':
        errs = [np.random.normal(0, stdev, (p,)) for _ in range(length)]
    elif noise_type == 'T':
        #print('yes')
        errs = [np.random.standard_t(5, (p,)) for _ in range(length)]
    ts = np.zeros((num_obs, p))
    for i in range(0, num_obs):
        ts[i,:] = ma_help(weight_matrices, errs, stdev)
    return ts



def calculate_auto_variance(ts, lag):
    n, p = ts.shape
    gamma_lag = np.zeros((p,p))
    if lag >= 0:
        for t in range(lag, n):
            gamma_lag += np.outer(ts[int(t),:], ts[int(t-lag),:])
    else:
        for t in range(-lag, n):
            gamma_lag += np.outer(ts[int(t),:], ts[int(t+lag),:])
    return gamma_lag/n



def one_var_autocovariance(weight, lag, stdev, N=30):
    if lag<0:
        return np.transpose(one_var_autocovariance(weight, -lag, stdev))
    p, _ = weight.shape
    gamma_lag = np.zeros((p,p))
    right_B = np.diag(np.repeat(1., p))
    left_B = np.linalg.matrix_power(weight, lag)
    for _ in range(lag, N):
        gamma_lag += np.dot(left_B, right_B.T)
        left_B = np.dot(left_B, weight)
        right_B = np.dot(right_B, weight)
    return gamma_lag*(stdev**2)



def ma_autocovariance(weights, lag, stdev):
    if lag<0:
        return np.transpose(ma_autocovariance(weights, -lag, stdev))
    length = len(weights)
    p, _ = weights[0].shape
    autocovariance = np.zeros((p,p))
    if lag>=length:
        return np.zeros((p,p))
    for ell in range(lag, length):
        autocovariance += np.dot(weights[ell], np.transpose(weights[ell-lag]))
    return autocovariance*(stdev**2)



def iid_autocovariance(lag, p, stdev):
    if lag == 0:
        return np.diag(np.repeat(stdev**2, p))
    return np.zeros((p,p))



def uni_auto_covariance(ts, lag):
    res = 0
    if lag>=0:
        for t in range(lag, len(ts)):
            res += ts[t]*ts[t-lag]
    elif lag<0:
        for t in range(0, len(ts)+lag):
            res += ts[t]*ts[t-lag]
    return res/(len(ts)-lag)



def ma_uni_help(weights, noise_arr, stdev):
    length = len(weights)
    value = 0
    for i in range(length - 1, -1, -1):
        value += np.dot(weights[length - i - 1], noise_arr[i])
    noise_arr.pop(0)
    noise_arr.append(np.random.normal(0, stdev, 1))
    return value


def generate_uni_ma(weight_list, num_obs, stdev, noise_type='G'):
    length = len(weight_list)
    if noise_type == 'G':
        errs = [np.random.normal(0, stdev, 1) for _ in range(length)]
    elif noise_type == 'T':
        errs = [np.random.standard_t(5, 1) for _ in range(length)]
    ts = np.zeros((num_obs, ))
    for i in range(0, num_obs):
        ts[i] = ma_uni_help(weight_list, errs, stdev)
    return ts


def generate_uni_var(transition_list, num_obs, stdev, starting_vec=None, noise_type='G'):
    lag = len(transition_list)
    if starting_vec is None:
        starting_vec = np.random.normal(0, stdev, lag)
    ts = np.zeros((num_obs, ))
    ts[0:lag] = starting_vec
    for i in range(lag, num_obs):
        for item in range(0, lag):
            ts[i] += np.dot(transition_list[item], ts[i - item - 1])
        if noise_type == 'G':
            errs = np.random.normal(0, stdev, 1)
        elif noise_type == 'T':
            errs = np.random.standard_t(5, 1)
        ts[i] += errs
    return ts





if __name__ == "__main__":

    ts = generate_uni_ma([1,0.8], 200, 1,  noise_type='G')
    print(ts)
    gamma_1 = uni_auto_covariance(ts, 1)
    print(gamma_1)
    gamma_0 = uni_auto_covariance(ts, 0)
    print(gamma_0)
    gamma_2 = uni_auto_covariance(ts, 2)
    print(gamma_2)




