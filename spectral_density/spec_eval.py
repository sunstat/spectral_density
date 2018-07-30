import numpy as np
from utilities import relative_err
from utilities import HS_norm

class SpecEval(object):

    @staticmethod
    def query_error(est_spec, true_spec, freq_index):
        '''
        :param est_spec: estimator dictionary
        :param true_spec: true dictionary
        :param relative_err: return relative or not
        :return: err
        '''
        return HS_norm(est_spec[freq_index] - true_spec[freq_index])**2

    @staticmethod
    def query_errors(est_spec, true_spec):
        '''
        :param est_spec: dict from freq_ind to estimated spec
        :param true_spec: dict from freq_ind to true spec
        :param relative_err_flag:
        :return:
        '''
        mse_dict = {}
        for freq_ind in true_spec.keys():
            mse = SpecEval.query_error(est_spec, true_spec, freq_ind)
            #print(err)
            mse_dict[freq_ind] = mse
        return mse_dict


    @staticmethod
    def get_three_metrics(true_spectral, spectral_estimator, individual_level=True, tolerance = 1e-20):

        if individual_level:
            precisions = []
            recalls = []
            F1s = []
            for freq_ind in true_spectral.keys():
                true_signal = abs(true_spectral[freq_ind])
                estimator = abs(spectral_estimator[freq_ind])
                ind = np.diag_indices(true_spectral[freq_ind].shape[0])
                true_signal[ind] = 0
                estimator[ind] = 0
                A1 = np.sum(np.logical_and(true_signal>tolerance, estimator>tolerance))
                A2 = np.sum(true_signal>tolerance)
                A3 = np.sum(estimator>tolerance)
                if A3 == 0:
                    precision = 1
                else:
                    precision = A1/A3
                recall = A1/A2
                precisions.append(precision)
                recalls.append(recall)
                F1s.append(2*precision*recall/(precision+recall))
            precision = np.mean(precisions)
            recall = np.mean(recalls)
            F1 = np.mean(F1s)
        else:
            true_signals = np.empty(true_spectral.shape)
            estimator_signals = np.empty(true_spectral.shape)
            correct_findings = np.empty(true_spectral.shape)
            true_signals.fill(False)
            estimator_signals.fill(False)
            correct_findings.fill(False)
            for freq_ind in true_spectral.keys():
                true_signal = abs(true_spectral[freq_ind])
                estimator_signal = abs(spectral_estimator[freq_ind])
                ind = np.diag_indices(true_spectral[freq_ind].shape[0])
                true_signal[ind] = 0
                estimator_signal[ind] = 0
                true_signals = np.logical_or(true_signals, true_signal>tolerance)
                estimator_signals = np.logical_or(estimator_signals, estimator_signal>tolerance)

            A1 = np.sum(np.logical_and(true_signals > tolerance, estimator_signals > tolerance))
            A2 = np.sum(true_signals > tolerance)
            A3 = np.sum(estimator_signals > tolerance)

            if A3 == 0:
                precision = 1
            else:
                precision = A1/A3
                recall = A1/A2

        return precision, recall, F1


if __name__ == "__main__":
    pass




