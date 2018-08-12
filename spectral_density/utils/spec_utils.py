from matplotlib import pyplot
import numpy as np
from utilities import *


def generate_dis_Fourier_freq_index(n):
    low_bound = int(np.ceil(-(n - 1) / 2.0))
    upper_bound = n / 2
    return [int(item) for item in np.arange(int(low_bound), int(upper_bound) + 1, 1)]


def index_to_freq(ind, n):
    return 2*np.pi*ind/n



def generate_dis_Fourier_coefs(freq, n):
    #print("frequency is {}".format(freq))
    return np.exp(-1j*freq * np.arange(0, n)) / np.sqrt(n)



def get_periodogram(ts, freq_index):
    num_obs, p = ts.shape
    freq = index_to_freq(freq_index, num_obs)
    df = np.dot(np.transpose(ts), generate_dis_Fourier_coefs(freq, num_obs))
    periodogram = 1/(2*np.pi)*np.outer(df, np.conj(df))
    return periodogram




def query_true_spectral(model, coefs, freq, stdev):
    if model == 'ma':
        A = coefs[0]
        A = A.astype(np.complex)
        for ell in range(1, len(coefs)):
            coef = coefs[ell]
            coef = coef.astype(np.complex)
            A += coef * np.exp(-1j * ell*freq)
        return 1 / (2 * np.pi) * np.dot(A, np.transpose(np.conj(A))) * (stdev ** 2)
    elif model == 'var':
        p, _ = coefs[0].shape
        A = np.diag(np.repeat(1.0 + 0.0j, p))
        for ell in range(len(coefs)):
            A -= coefs[ell] * np.exp(-1j * (ell + 1)*freq)
        return 1 / (2 * np.pi) * np.dot(np.linalg.inv(A), np.transpose(np.conj(np.linalg.inv(A)))) * (stdev ** 2)




if __name__ == "__main__":

    from mults_utils import *
    print("test discrete Fourier indices")
    print(generate_dis_Fourier_freq_index(11))
    print(generate_dis_Fourier_freq_index(10))
    print("========")
    print("test index to frequency")
    print(index_to_freq(1, 11))
    print(2*np.pi*1/11)
    print("========")
    print("test discrete Fourier Coeficients")
    w = generate_dis_Fourier_coefs(index_to_freq(1, 11), 11)
    print(w)
    print(np.dot(w, np.conj(w)))

    print(generate_dis_Fourier_coefs(0, 100))

    stdev = 1
    num_obs = 800
    span = 20
    B1 = np.array([[0.5, 0.0], [0.0, 0.2]])
    weights = [np.diag(np.repeat(1., 2)), B1]
    ts = generate_ma(weights, num_obs, stdev)
    print("begin of printing peridogram")
    peridogram = get_periodogram(ts, 40)
    print("========")
    print(peridogram)
    peridogram = get_periodogram(ts, 0)
    print(peridogram)
    df = np.dot(ts.T, np.repeat(1, num_obs))
    print(1 / (2 * np.pi) * np.outer(df, df)/num_obs)
    print("end of printing peridogram")
    print(query_true_spectral('ma',weights, 0, 1))
