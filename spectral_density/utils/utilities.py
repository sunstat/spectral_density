import numpy as np
import random
from functools import reduce
#from mults_utils import *

'''
base help functions
'''

QUIET = True


'''
linear algebra session 
'''
def standadize_matrix(mat, standard_const):
    return mat/np.linalg.norm(mat, 'fro')*standard_const


def HS(mat1, mat2):
    p, _ = mat1.shape
    res = np.trace(np.dot(mat1, np.transpose(np.conj(mat2))))/p
    return np.real(res)


def HS_norm(mat):
    return np.sqrt(HS(mat, mat))


def gene_scheme(model, coefs, span):
    return {'model': model, 'coefs': coefs, 'span':span}


def diff(mat1, mat2):
    p, _ = mat1.shape
    return HS_norm(mat1-mat2)

def relative_err(mat, target):
    return HS_norm(mat-target)/HS_norm(target)


'''
smoothing/split smoothing session
'''
def decrement_one(cur_ind, ls_keys):
    if cur_ind == np.min(ls_keys):
        return np.max(ls_keys)
    return (cur_ind-1)

def increment_one(cur_ind, ls_keys):
    if cur_ind == np.max(ls_keys):
        return np.min(ls_keys)
    return cur_ind+1

def generate_neighobors(cur_index, ls_keys, span):
    ls = []
    ls.append(cur_index)
    left_ind = cur_index
    right_ind = cur_index
    for _ in range(span):
        left_ind = decrement_one(left_ind, ls_keys)
        right_ind = increment_one(right_ind, ls_keys)
        ls.append(left_ind)
        ls.append(right_ind)
    return tuple(ls)


def sample_split(neighbors):
    neigh = list(neighbors)
    left_ls = []
    right_ls = []
    flag = True
    while len(neigh)>0:
        values = []
        ind = random.randint(0, len(neigh)-1)
        value = neigh[ind]
        values.append(value)
        if -value in neigh and value != 0:
            values.append(-value)
            neigh.remove(-value)
        neigh.remove(value)
        if flag :
            left_ls.extend(values)
        else:
            right_ls.extend(values)
        flag = not flag
    return left_ls, right_ls


def smooth_subindex(dict_matrices, index_set):
    res = np.copy(dict_matrices[index_set[0]])
    for i in range(1, len(index_set)):
        res += dict_matrices[index_set[i]]
    return res/ len(index_set)


'''
operators session: thresholding 
'''
def hard_threshold_operator(spd, threshold_value, diag_flag = True):
    res = np.copy(spd)
    res[abs(res) < threshold_value] = 0 + 0j
    if diag_flag:
        ind= np.diag_indices(res.shape[0])
        res[ind] = spd[ind]
    return res

def soft_threshold_operator(spd, threshold_value, diag_flag = True):
    res = np.copy(spd)
    res_sd = res/abs(res)
    right_part = abs(res)-threshold_value
    right_part[right_part<0] = 0
    res = res_sd*right_part
    if diag_flag:
        for i in range(spd.shape[0]):
            res[i,i] = spd[i,i]
    return res

def adaptive_lasso_operator(spd, threshold_value, eta=2, diag_flag=True):
    res = np.copy(spd)
    res_sd = res / abs(res)
    right_part = abs(res) - threshold_value**(eta+1)/(abs(res)**eta)
    right_part[right_part<0] = 0
    res =  res_sd * right_part
    if diag_flag:
        for i in range(spd.shape[0]):
            res[i,i] = spd[i,i]
    return res


def optimal_general_thresholding_estimator(dict_matrices, neighbors, smooth_estimator, threshold_operator, num_grid=50):
    left_ls, right_ls = sample_split(neighbors)
    base_spd = smooth_subindex(dict_matrices, left_ls)
    threshold_psd = smooth_subindex(dict_matrices, right_ls)
    discrepency = np.float('Inf')
    threshold_value = 0
    dif = abs(threshold_psd-np.diag(np.diag(threshold_psd)))
    for value in np.linspace(start=np.min(dif), stop=np.max(dif), num=num_grid):
        candidate_threshold = threshold_operator(threshold_psd, value)
        err = HS_norm(base_spd - candidate_threshold)
        if discrepency > err:
            discrepency = err
            threshold_value = value
    '''
    if abs(threshold_value-np.max(abs(threshold_psd-np.diag(np.diag(threshold_psd))))) < 1e-10:
        print(candidate_threshold)
    '''
    threshold_estimator = threshold_operator(smooth_estimator, threshold_value)
    # print(threshold_value)
    return threshold_estimator



def smooth_matrices_single_index(cur_ind, dict_matrices, ls_keys, span):
    res = np.copy(dict_matrices[cur_ind])
    count = 0
    left_index = cur_ind
    right_index = cur_ind
    while count<span:
        #print("count is {}".format(count))
        #print(ls_keys)
        left_index = decrement_one(left_index, ls_keys)
        right_index = increment_one(right_index, ls_keys)
        res += dict_matrices[left_index]
        res += dict_matrices[right_index]
        count+=1
    return res/(2*span+1)


def smooth_matrices(dict_matrices, ls_keys, span):
    smoothing_dict_matrices = {}
    cur_index = 0
    smoothing_dict_matrices[cur_index] = smooth_matrices_single_index(cur_index, dict_matrices, ls_keys, span)
    left_index = -span
    right_index = span
    count = 1
    length = 2*span+1
    value = smoothing_dict_matrices[cur_index] * length
    while count<len(ls_keys):
        value -= dict_matrices[left_index]
        left_index = increment_one(left_index, ls_keys)
        right_index = increment_one(right_index, ls_keys)
        value += dict_matrices[right_index]
        cur_index = increment_one(cur_index, ls_keys)
        #print("current index is {}, left index is {} and right index is {}".format(cur_index, left_index, right_index))
        smoothing_dict_matrices[cur_index] = value/length
        count += 1
        if not QUIET and count%50 == 0:
            print("finishing smoothing {}".format(count))
    return smoothing_dict_matrices


'''
average of errs_dict
'''
def average_errs_dict(ls_dict, key_ls):
    err_dict = {}
    for key in key_ls:
        err_dict[key] = np.mean(list(map(lambda x: x[key], ls_dict)))
    return err_dict


def abs_pass(my_dict):
    for key in my_dict:
        my_dict[key] = abs(my_dict[key])


def average_abs_dict(my_dict):
    abs_pass(my_dict)
    ls = list(my_dict.values())
    return 1.0*sum(ls)/len(my_dict)

def l2_average_abs_dict(my_dict):
    abs_pass(my_dict)
    ls = list(np.asarray(list(my_dict.values()))**2)
    return 1.0*sum(ls)/len(my_dict)


def max_abs_dict(my_dict):
    abs_pass(my_dict)
    ls = list(my_dict.values())
    res = np.dstack(ls).max(axis=2)
    return res


def cohenrance(mat, flag = True):
    if flag:
        D = np.real(np.diag(1.0/np.sqrt(np.diag(mat))))
        res = reduce(np.dot, [D, mat, D])
        res[np.diag_indices(res.shape[0])] = 0
    else:
        res = mat
    return res


def coherance_pass(my_dict):
    for key in my_dict:
        my_dict[key] = cohenrance(abs(my_dict[key]))


def average_abs_coherance(my_dict):
    coherance_pass(my_dict)
    ls = list(my_dict.values())
    return sum(ls)/len(ls)


if __name__ == "__main__":
    pass