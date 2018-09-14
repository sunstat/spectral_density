import numpy as np
from utils.utilities import *
from utils.mults_utils import *
from utils.spec_utils import *
from spectral_density.spec_eval import SpecEval
from generating_weights import *

def deviance(f_1, f_2, span):
    #print(f_1)
    #print(f_2)
    #print(span)
    #return -np.log(f_1 / f_2) + (f_1 - f_2) / f_2
    return (-np.log(f_1/f_2)+(f_1-f_2)/f_2)/((1-1/(2*span+1))**2)



class uni_spec_est(object):

    def __init__(self, ts, model_info, simu=True):
        self.ts = ts
        self.num_obs = len(self.ts)
        self.frequency_indices = generate_dis_Fourier_freq_index(self.num_obs)
        self.model = model_info['model']
        self.weights = model_info['weights']
        self.stdev = model_info['stdev']
        self.span = model_info['span']
        self.smoothing_estimator = {}
        self.periodograms = {}
        self.true_spectral = {}
        self.get_periodograms()
        self._smooth(self.span)
        if simu:
            self._get_true_spectral()



    def get_periodograms(self):
        for freq_index in self.frequency_indices:
            self.periodograms[freq_index] = get_uni_periodogram(self.ts, freq_index)
            # print(self.periodograms[-49].shape)


    def _smooth(self, span):
        '''
        :param span: assuming span is less than n/2
        :return: None, but update the value of self.smooth_periodograms
        '''
        self.smoothing_estimator = smooth_scalars(self.periodograms, self.frequency_indices, span)
        self.span = span
        #print(self.smoothing_estimator[0])


    def query_periodogram(self, freq_index):
        return self.periodograms[freq_index]


    def query_smoothing_estimator(self, freq_index):
        return self.smoothing_estimator[freq_index]


    def query_true_spectral(self, freq_index):
        return self.true_spectral[freq_index]


    def _get_true_spectral(self):
        for freq_ind in self.frequency_indices:
            self.true_spectral[freq_ind] = \
                query_uni_true_spectral(self.model, self.weights, index_to_freq(freq_ind, self.num_obs), self.stdev)


    def spec_dev(self):
        dev = 0
        #print(self.span)
        for freq_ind in self.frequency_indices:
            if freq_ind>=0:
                if freq_ind == 0:
                    #print(self.span)
                    dev += 0.5*deviance(self.query_periodogram(freq_ind), self.query_smoothing_estimator(freq_ind), self.span)
                else:
                    dev += deviance(self.query_periodogram(freq_ind), self.query_smoothing_estimator(freq_ind), self.span)
        return dev


    def find_optimal_span(self):
        optimal_span = None
        opt_dev = np.inf
        for span in range(1, self.num_obs//2):
            self._smooth(span)
            dev = self.spec_dev()
            print('span is {} and dev is {}'.format(span, dev))
            if opt_dev>dev:
                opt_dev = dev
                optimal_span = span
        return optimal_span



if  __name__ == "__main__":
    '''
    weight_list = [0.2,0.5]
    num_obs = 200
    stdev = 1
    noise_type = 'G'
    span = 20
    errs = []
    for _ in range(1):
        ts = generate_uni_var(weight_list, num_obs, stdev, noise_type='G')
        model_info = {}
        model_info['model'] = 'ma'
        model_info['weights'] = weight_list
        model_info['span'] = span
        model_info['stdev'] = stdev

        sd = uni_spec_est(ts, model_info)
        print(sd.find_optimal_span())
        errs.append(np.abs(sd.query_true_spectral(0) - sd.query_smoothing_estimator(0))/sd.query_true_spectral(0))
        print(sd.find_optimal_span())
    '''
    stdev = 1
    num_obs = 100
    weights = fetch_weights(p=3, ho_heter_mode='ho', gen_model='ma')
    print(weights[1][0:3,0:3])
    ts = generate_ma(weights, num_obs, stdev, noise_type='G')
    model_info = {}
    model_info['model'] = 'ma'
    model_info['weights'] = weights
    model_info['span'] = 1
    model_info['stdev'] = stdev
    span_list=np.zeros((3,4))
    for t in range(4):
        ts = generate_ma(weights, num_obs, stdev, noise_type='G')
        for position in range(3):
            uni_spec = uni_spec_est(ts[:,position], model_info, simu=False)
            optimal_span = uni_spec.find_optimal_span()
            #print(optimal_span)
            span_list[position, t] = optimal_span

    print(span_list)










