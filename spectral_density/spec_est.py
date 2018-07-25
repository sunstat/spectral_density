import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))

from utils.utilities import *
from utils.mults_utils import *
from utils.spec_utils import *
from spectral_density.spec_eval import SpecEval


class SpecEst(object):

    def _get_neighbors(self):
        for freq_ind in self.frequency_indices:
            self.neighbors[freq_ind] = generate_neighobors(freq_ind, self.frequency_indices, self.span)

    def get_periodograms(self):
        for freq_index in self.frequency_indices:
            self.periodograms[freq_index] = get_periodogram(self.ts, freq_index)
            # print(self.periodograms[-49].shape)

    def _smooth(self):
        '''
        :param span: assuming span is less than n/2
        :return: None, but update the value of self.smooth_periodograms
        '''
        self.smoothing_estimator = smooth_matrices(self.periodograms, self.frequency_indices, self.span)

    def _get_three_metrics(self):
        self.precision['th'], self.recall['th'], self.F1['th'] \
            = SpecEval.get_three_metrics(self.true_spectral, self.thresholding_estimator)
        self.precision['so'], self.recall['so'], self.F1['so'] \
            = SpecEval.get_three_metrics(self.true_spectral, self.soft_threshold_estimator)
        self.precision['al'], self.recall['al'], self.F1['al'] \
            = SpecEval.get_three_metrics(self.true_spectral, self.adaptive_lasso_estimator)

    '''
    shrinkage session
    '''

    def _get_shrinkage_estimator(self, freq_index):
        spd = self.smoothing_estimator[freq_index]
        mu_hat = np.trace(spd) * 1.0 / self.p
        delta_sq_hat = HS_norm(spd - mu_hat * np.diag(np.repeat(1, self.p))) ** 2
        beta_sq_hat = HS_norm(self.periodograms[freq_index] - spd) ** 2
        length = 2 * self.span + 1
        left_index = decrement_one(freq_index, self.frequency_indices)
        right_index = increment_one(freq_index, self.frequency_indices)
        count = 0
        while count < self.span:
            beta_sq_hat += HS_norm(self.periodograms[left_index] - spd) ** 2
            beta_sq_hat += HS_norm(self.periodograms[right_index] - spd) ** 2
            left_index = decrement_one(freq_index, self.frequency_indices)
            right_index = increment_one(freq_index, self.frequency_indices)
            count += 1
        beta_sq_hat = np.min([beta_sq_hat / (length ** 2), delta_sq_hat])
        alpha_sq_hat = delta_sq_hat - beta_sq_hat
        return beta_sq_hat / delta_sq_hat * mu_hat * np.diag(np.repeat(1, self.p)) + alpha_sq_hat / delta_sq_hat * \
                                                                                     self.smoothing_estimator[
                                                                                         freq_index]

    def _get_shrinkage_estimators(self):
        for freq_ind in self.frequency_indices:
            self.shrinkage_estimator[freq_ind] = self._get_shrinkage_estimator(freq_ind)

    '''
    thresholding session
    '''

    def _get_thresholding_estimator(self):
        for freq_index in self.frequency_indices:
            self.thresholding_estimator[freq_index] = \
                optimal_general_thresholding_estimator(self.periodograms,
                                                       self.neighbors[freq_index], self.smoothing_estimator[freq_index],
                                                       hard_threshold_operator, num_grid=40)
            self.soft_threshold_estimator[freq_index] = optimal_general_thresholding_estimator(self.periodograms,
                                                                                               self.neighbors[
                                                                                                   freq_index],
                                                                                               self.smoothing_estimator[
                                                                                                   freq_index],
                                                                                               soft_threshold_operator,
                                                                                               num_grid=40)
            self.adaptive_lasso_estimator[freq_index] = optimal_general_thresholding_estimator(self.periodograms,
                                                                                               self.neighbors[
                                                                                                   freq_index],
                                                                                               self.smoothing_estimator[
                                                                                                   freq_index],
                                                                                               adaptive_lasso_operator,
                                                                                               num_grid=40)

    '''
    true spectral session
    '''

    def _get_true_spectral(self):
        for freq_ind in self.frequency_indices:
            self.true_spectral[freq_ind] = \
                query_true_spectral(self.model, self.weights, index_to_freq(freq_ind, self.num_obs),
                                             self.stdev)

    def _fetch_heat_maps(self):
        self.heat_map['th'] = {}
        self.heat_map['sh'] = {}
        self.heat_map['sm'] = {}
        self.heat_map['al'] = {}
        self.heat_map['true'] = {}
        freq_ind_set = [0, self.num_obs // 4, -(self.num_obs // 4)]
        '''
        for freq_ind in freq_ind_set:
            self.heat_map['th'][freq_ind] = cohenrance(abs(self.query_thresholding_estimator(freq_ind)))
            self.heat_map['sh'][freq_ind] = cohenrance(abs(self.query_shrinkage_estimator(freq_ind)))
            self.heat_map['sm'][freq_ind] = cohenrance(abs(self.query_smoothing_estimator(freq_ind)))
            self.heat_map['al'][freq_ind] = cohenrance(abs(self.query_adaptive_lasso_estimator(freq_ind)))
            if self.simu:
                self.heat_map['true'][freq_ind] = abs(self.query_true_spectral(freq_ind))
        '''

        self.heat_map['sm']['ave'] = average_abs_coherance(self.smoothing_estimator)
        self.heat_map['th']['ave'] = average_abs_coherance(self.thresholding_estimator)
        self.heat_map['sh']['ave'] = average_abs_coherance(self.shrinkage_estimator)
        self.heat_map['al']['ave'] = average_abs_coherance(self.adaptive_lasso_estimator)

        '''
        self.heat_map['sm']['ave_2'] = cohenrance(l2_average_abs_dict(self.smoothing_estimator))
        self.heat_map['th']['ave_2'] = cohenrance(l2_average_abs_dict(self.thresholding_estimator))
        self.heat_map['sh']['ave_2'] = cohenrance(l2_average_abs_dict(self.shrinkage_estimator))
        self.heat_map['al']['ave_2'] = cohenrance(l2_average_abs_dict(self.adaptive_lasso_estimator))


        self.heat_map['sm']['max'] = cohenrance(max_abs_dict(self.smoothing_estimator))
        self.heat_map['th']['max'] = cohenrance(max_abs_dict(self.thresholding_estimator))
        self.heat_map['sh']['max'] = cohenrance(max_abs_dict(self.shrinkage_estimator))
        self.heat_map['al']['max'] = cohenrance(max_abs_dict(self.adaptive_lasso_estimator))
        '''

        if self.simu:
            self.heat_map['true']['ave'] = cohenrance(average_abs_dict(self.true_spectral))

    def __init__(self, ts, model_info, simu=True):
        '''
        :param ts: sample time series
        :param gene_scheme: true generating scheme
        '''
        self.ts = ts
        self.model = model_info['model']
        self.weights = model_info['weights']
        self.stdev = model_info['stdev']
        self.span = model_info['span']
        self.num_obs, self.p = self.ts.shape
        self.frequency_indices = generate_dis_Fourier_freq_index(self.num_obs)
        # print(self.frequency_indices)
        self.periodograms = {}
        self.get_periodograms()
        self.freq_ls = generate_dis_Fourier_freq_index(self.num_obs)
        self.simu = simu
        '''
        define dictionary for 
        '''

        self.smoothing_estimator = {}
        self.shrinkage_estimator = {}
        self.thresholding_estimator = {}
        self.true_spectral = {}
        self.neighbors = {}
        self.soft_threshold_estimator = {}
        self.adaptive_lasso_estimator = {}
        self._smooth()
        self._get_shrinkage_estimators()
        if self.simu:
            self._get_true_spectral()
        self._get_neighbors()
        self._get_thresholding_estimator()
        # for heat maps
        self.heat_map = {}
        self._fetch_heat_maps()
        self._get_true_spectral()
        # precision, recall, F1
        self.precision = {}
        self.recall = {}
        self.F1 = {}
        self._get_three_metrics()

    '''
    query section
    '''

    def query_smoothing_estimator(self, freq_index):
        return self.smoothing_estimator[freq_index]

    def query_shrinkage_estimator(self, freq_index):
        return self.shrinkage_estimator[freq_index]

    def query_thresholding_estimator(self, freq_index):
        return self.thresholding_estimator[freq_index]

    def query_periodogram(self, freq_index):
        return self.periodograms[freq_index]

    def query_true_spectral(self, freq_index):
        return self.true_spectral[freq_index]

    def query_peridogram(self, freq_index):
        return self.periodograms[freq_index]

    def query_neighbors(self, freq_index):
        return self.neighbors[freq_index]

    def query_soft_threshold_estimator(self, freq_index):
        return self.soft_threshold_estimator[freq_index]

    def query_adaptive_lasso_estimator(self, freq_index):
        return self.adaptive_lasso_estimator[freq_index]

    def return_all_periodograms(self):
        return self.periodograms

    def return_all_true_spectral(self):
        return self.true_spectral

    def query_heat_map(self):
        return self.heat_map

    def query_recover_three_measures(self, mode):
        assert mode in ['th', 'so', 'al']
        return self.precision[mode], self.recall[mode], self.F1[mode]

    def evaluate(self, mode, sampling_size=-1):
        err_dict = None
        if mode == 'sm':
            err_dict = SpecEval.average_err(self.smoothing_estimator, self.true_spectral, sampling_size=sampling_size)
        if mode == 'sh':
            err_dict = SpecEval.average_err(self.shrinkage_estimator, self.true_spectral, sampling_size=sampling_size)
        if mode == 'th':
            err_dict = SpecEval.average_err(self.thresholding_estimator, self.true_spectral,
                                            sampling_size=sampling_size)
        if mode == 'al':
            err_dict = SpecEval.average_err(self.adaptive_lasso_estimator, self.true_spectral,
                                            sampling_size=sampling_size)
        if mode == 'so':
            err_dict = SpecEval.average_err(self.soft_threshold_estimator, self.true_spectral,
                                            sampling_size=sampling_size)
        return err_dict


if __name__ == "__main__":
    stdev = 0.1
    num_obs = 800
    span = 20
    B1 = np.array([[0.0, 0.8], [0.7, .0]])
    weights = [np.diag(np.repeat(1., 2)), B1]
    model_info = {}
    model_info['model'] = 'ma'
    model_info['weights'] = weights
    model_info['stdev'] = stdev
    model_info['span'] = span
    errs = []
    ts = generate_ma(weights, num_obs=num_obs, stdev=stdev)
    print(calculate_auto_variance(ts, 0))
    print((np.dot(B1, B1.T) + np.diag(np.repeat(1., 2))) * stdev ** 2)

    model_info = {}
    model_info['model'] = 'ma'
    model_info['weights'] = weights
    model_info['span'] = 5
    model_info['stdev'] = stdev
    ts = generate_ma(weights, num_obs=num_obs, stdev=stdev)
    spec_est = SpecEst(ts, model_info)
    print(spec_est.query_neighbors(0))
    print(spec_est.query_neighbors(3))