import numpy as np


def index_to_freq(ind, n):
    return 2*np.pi*ind/n



def generate_mvar(transition_matrices, num_obs, stdev, starting_vec=None):
    lag = len(transition_matrices)
    p, _ = transition_matrices[0].shape
    if starting_vec is None:
        starting_vec = np.random.normal(0, stdev, (lag, p))
    ts = np.zeros((num_obs, p))
    ts[0:lag, :] = starting_vec
    for i in range(lag, num_obs):
        for item in range(0, lag):
            ts[i, :] += np.dot(transition_matrices[item], ts[i - item - 1, :])
        ts[i, :] += np.random.normal(0, stdev, (p,))
    return ts



def ma_help(weights, noise_arr, stdev):
    '''
    :param weights: weights for the noise
    :param noise_arr: noise_array
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


def generate_ma(weight_matrices, num_obs, stdev):
    length = len(weight_matrices)
    p, _ = weight_matrices[0].shape
    errs = [np.random.normal(0, stdev, (p,)) for _ in range(length)]
    ts = np.zeros((num_obs, p))
    for i in range(0, num_obs):
        ts[i,:] = ma_help(weight_matrices, errs, stdev)
    return ts


def calculate_auto_variance(ts, lag):
    n, p = ts.shape
    gamma_lag = np.zeros((p,p))
    if lag >= 0:
        for t in range(0, n-lag):
            gamma_lag += np.outer(ts[int(t),:], ts[int(t+lag),:])
    else:
        for t in range(-lag, n):
            gamma_lag += np.outer(ts[int(t),:], ts[int(t+lag),:])
    return gamma_lag/n

def get_periodogram_naive(ts, freq_index):
    n, p = ts.shape
    periodogram = np.repeat(0.0+0.0j, p*p).reshape(p, p)
    for lag in range(-(n-1), (n-1), 1):
        periodogram += MulTS.calculate_auto_variance(ts, lag)*np.exp(-1j*lag*MulTS.index_to_freq(freq_index, n))
    return 1/(2*np.pi)*periodogram


def one_var_autocovariance(weight, lag, stdev, N=30):
    if lag<0:
        return np.transpose(MulTS.one_var_autocovariance(weight, -lag, stdev))
    p, _ = weight.shape
    gamma_lag = np.zeros((p,p))
    left_B = np.diag(np.repeat(1., p))
    right_B = np.linalg.matrix_power(weight, lag)
    for _ in range(lag, N):
        gamma_lag += np.dot(left_B, right_B.T)
        left_B = np.dot(left_B, weight)
        right_B = np.dot(right_B, weight)
    return gamma_lag*(stdev**2)


def ma_autocovariance(weights, lag, stdev):
    if lag<0:
        return np.transpose(MulTS.ma_autocovariance(weights, -lag, stdev))
    length = len(weights)
    p, _ = weights[0].shape
    autocovariance = np.zeros((p,p))
    if lag>=length:
        return np.zeros((p,p))
    for ell in range(0, length-lag):
        autocovariance += np.dot(weights[ell], np.transpose(weights[ell+lag]))
    return autocovariance*(stdev**2)

def iid_autocovariance(lag, p, stdev):
    if lag == 0:
        return np.diag(np.repeat(stdev**2, p))
    return np.zeros((p,p))


if __name__ == "__main__":
    pass




